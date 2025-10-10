"""
Convert circuit to graph (PyG's Data) and vice versus.
"""

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import torch

from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from utils import GATE_CLS_MAP, get_qubit_index


def dag_to_pyg_data(dag: DAGCircuit,
                    num_qubits: int,
                    basic_gates: list
                    ) -> Data:
    """将Qiskit DAG转换为PyG的Data对象（节点特征：门类型+比特掩码）"""
    gate_type_map = {g: i for i, g in enumerate(basic_gates)}
    op_nodes = [node for node in dag.nodes() if isinstance(node, DAGOpNode)]
    node_features = []

    # 构建节点特征（门类型独热编码 + 比特掩码）
    for node in op_nodes:
        gate_name = node.op.name.lower()
        if gate_name not in gate_type_map:
            continue  # 跳过不支持的门（如测量门）

        # 门类型特征（独热编码）
        gate_feat = torch.zeros(len(basic_gates))
        gate_feat[gate_type_map[gate_name]] = 1.0

        # 比特掩码特征（标记门作用的比特）
        qubit_feat = torch.zeros(num_qubits)
        for q in node.qargs:
            q_idx = get_qubit_index(q)
            if 0 <= q_idx < num_qubits:  # 过滤无效索引
                qubit_feat[q_idx] = 1.0

        node_features.append(torch.cat([gate_feat, qubit_feat]))

    # 构建边索引（DAG的依赖关系）
    edge_index = []
    node_idx_map = {node: i for i, node in enumerate(op_nodes)}
    for i, node in enumerate(op_nodes):
        for pred_node in dag.predecessors(node):
            if pred_node in node_idx_map:
                edge_index.append([node_idx_map[pred_node], i])

    # 处理空电路情况
    if not node_features:
        return Data(
            x=torch.empty((0, len(basic_gates) + num_qubits)),
            edge_index=torch.empty((2, 0), dtype=torch.long)
        )

    return Data(
        x=torch.stack(node_features),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    )

def data_to_quantum_circuit(data: Data,
                            num_qubits: int,
                            gate_types: list) -> QuantumCircuit:
    """
    从PyG的Data对象恢复量子电路（使用NetworkX简化拓扑排序）

    参数:
        data: PyG的Data对象（包含节点特征x和边索引edge_index）
        num_qubits: 量子电路的比特数
        gate_types: 门类型列表（与节点特征中的独热编码对应）

    返回:
        恢复的QuantumCircuit对象
    """
    # 初始化量子电路
    qc = QuantumCircuit(num_qubits)
    num_gate_types = len(gate_types)

    # 处理空电路
    if data.x is None or data.x.numel() == 0:
        return qc

    # 1. 解析节点特征：提取每个节点的门类型和作用比特
    node_features = data.x.cpu().numpy()
    gates = []  # 存储 (门名称, 作用比特列表)
    for feat in node_features:
        # 门类型（独热编码取最大值索引）
        gate_idx = feat[:num_gate_types].argmax()
        gate_name = gate_types[gate_idx]
        # 作用比特（掩码中值为1的索引）
        qubits = [i for i in range(num_qubits) if feat[num_gate_types + i] == 1.0]
        gates.append((gate_name, qubits))

    # 2. 确定门的执行顺序（使用NetworkX拓扑排序）
    if data.edge_index is not None and data.edge_index.numel() > 0:
        # 转换为NetworkX有向图（保持节点索引一致）
        nx_graph = to_networkx(data, to_undirected=False)  # 必须为有向图
        # 拓扑排序（返回节点索引的执行顺序）
        try:
            gate_order = list(nx.topological_sort(nx_graph))
        except nx.NetworkXUnfeasible:
            # 若图有环（理论上DAG不应有环），退化为节点顺序
            print("警告：图中存在环，使用节点顺序作为门顺序")
            raise
            # gate_order = list(range(len(gates)))
    else:
        # 无边缘信息，默认按节点顺序添加
        gate_order = list(range(len(gates)))

    # 3. 按顺序添加门到电路
    for node_idx in gate_order:
        gate_name, qubits = gates[node_idx]
        gate = GATE_CLS_MAP[gate_name]()
        # 验证比特数量是否符合门类型（单/双比特门）
        if len(qubits) == 1 and gate_name in ['h', 'x', 'z', 't', 's']:
            qc.append(gate, [qubits[0]])
        elif len(qubits) == 2 and gate_name in ['cx', 'swap']:
            qc.append(gate, qubits)  # 注意：CX门顺序是(控制, 目标)
        else:
            print(f"警告：不支持的门类型或比特数 {gate_name} {qubits}，已跳过")

    return qc
