import os
import random
from typing import List, Tuple, Dict, Optional
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.converters import circuit_to_dag
# 导入支持的门类
from qiskit.circuit.library import HGate, XGate, CXGate, SwapGate, ZGate, TGate, SGate
from tqdm import trange


def _get_qubit_index(q) -> int:
    """
    获取量子比特的索引，兼容不同Qiskit版本

    参数:
        q: Qubit对象

    返回:
        量子比特的整数索引
    """
    # 尝试多种获取索引的方式，确保兼容性
    if hasattr(q, 'index'):
        return q.index
    elif hasattr(q, '_index'):
        return q._index
    elif hasattr(q, 'register') and hasattr(q.register, 'qubits'):
        return q.register.qubits.index(q)
    else:
        raise AttributeError(f"无法获取量子比特索引，Qubit对象属性: {dir(q)}")


class QuantumCircuitDataset(Dataset):
    """
    量子电路优化数据集，支持通过basic_gates动态配置门类型
    """
    def __init__(self,
                 root: str = "./data",
                 num_samples: int = 1000,
                 num_qubits: int = 3,
                 max_depth: int = 10,
                 topo_type: str = "linear",
                 basic_gates: List[str] = None,  # 门类型字符串列表
                 regenerate: bool = False):
        """
        初始化数据集，支持动态门类型配置

        参数:
            basic_gates: 门类型字符串列表，如['h', 'x', 'cx', 'swap']
                         若为None则使用默认门集
        """
        super().__init__(root)

        # 设置默认门集
        self.basic_gates = basic_gates if basic_gates is not None else ['h', 'x', 'cx', 'swap']

        # 验证门类型有效性并建立映射关系
        self.gate_class_map = self._create_gate_class_map()
        self.single_qubit_gates, self.two_qubit_gates = self._split_gate_types()

        self.num_samples = num_samples
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.topo_type = topo_type
        self.coupling_map = self._generate_coupling_map()

        # self.processed_dir = os.path.join(root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        if regenerate or not self._check_processed_exists():
            self._generate_dataset()


    def _create_gate_class_map(self) -> Dict[str, type]:
        """
        建立门名称字符串到门类的映射

        返回:
            门名称到门类的字典，如{'h': HGate, 'x': XGate}
        """
        # 支持的门类型映射表，可根据需要扩展
        supported_gates = {
            'h': HGate,
            'x': XGate,
            'z': ZGate,
            't': TGate,
            's': SGate,
            'cx': CXGate,
            'swap': SwapGate,
        }

        # 验证输入的门类型是否都被支持
        for gate_name in self.basic_gates:
            if gate_name not in supported_gates:
                raise ValueError(f"不支持的门类型: {gate_name}，支持的门类型为: {list(supported_gates.keys())}")

        # 只返回用户指定的门类型映射
        return {gate_name: supported_gates[gate_name] for gate_name in self.basic_gates}

    def _split_gate_types(self) -> Tuple[List[str], List[str]]:
        """
        区分单量子比特门和双量子比特门

        返回:
            单量子比特门名称列表和双量子比特门名称列表
        """
        # 已知的双量子比特门
        two_qubit_gate_names = ['cx', 'swap']

        single = []
        two = []

        for gate_name in self.basic_gates:
            if gate_name in two_qubit_gate_names:
                two.append(gate_name)
            else:
                single.append(gate_name)

        if not single:
            raise ValueError("至少需要指定一种单量子比特门")

        return single, two

    def _check_processed_exists(self) -> bool:
        """检查处理好的数据是否已存在"""
        required_files = [
            f"sample_{i}.pt" for i in range(self.num_samples)
        ]
        return all(
            os.path.exists(os.path.join(self.processed_dir, f))
            for f in required_files
        )

    def _generate_coupling_map(self) -> CouplingMap:
        """生成物理拓扑对应的耦合图"""
        if self.topo_type == "linear":
            edges = [(i, i+1) for i in range(self.num_qubits - 1)]
            edges += [(i+1, i) for i in range(self.num_qubits - 1)]
        elif self.topo_type == "star":
            edges = [(0, i) for i in range(1, self.num_qubits)]
            edges += [(i, 0) for i in range(1, self.num_qubits)]
        else:
            raise ValueError(f"不支持的拓扑类型: {self.topo_type}")

        return CouplingMap(edges)

    def _generate_random_circuit(self) -> QuantumCircuit:
        """生成随机量子电路作为原始电路qc，使用指定的门类型"""
        qc = QuantumCircuit(self.num_qubits)

        for _ in range(random.randint(1, self.max_depth)):
            # 随机选择单比特门或双比特门（双比特门需要至少2个量子比特且存在可用的双比特门）
            use_two_qubit = (
                random.random() < 0.6 and
                self.num_qubits >= 2 and
                len(self.two_qubit_gates) > 0
            )

            if use_two_qubit:
                # 双比特门
                q1, q2 = random.sample(range(self.num_qubits), 2)
                gate_name = random.choice(self.two_qubit_gates)
                gate = self.gate_class_map[gate_name]()  # 实例化门
                qc.append(gate, [q1, q2])
            else:
                # 单比特门
                q = random.choice(range(self.num_qubits))
                gate_name = random.choice(self.single_qubit_gates)
                gate = self.gate_class_map[gate_name]()  # 实例化门
                qc.append(gate, [q])

        return qc

    @staticmethod
    def circuit_to_dag(qc: QuantumCircuit) -> DAGCircuit:
        """将量子电路转换为DAG图"""
        return circuit_to_dag(qc)

    def dag_to_data(self, dag: DAGCircuit, num_qubits: int) -> Data:
        """
        将DAG转换为PyTorch Geometric的Data对象，支持动态门类型

        节点特征:
            - 前N维: 门类型独热编码（N为门类型数量）
            - 后num_qubits维: 门操作的量子比特掩码
        """
        # 门类型映射（根据当前配置的basic_gates动态生成）
        gate_type_map = {gate_name: idx for idx, gate_name in enumerate(self.basic_gates)}
        num_gate_types = len(self.basic_gates)

        # 收集节点特征
        node_features = []
        op_nodes = [node for node in dag.nodes() if isinstance(node, DAGOpNode)]

        for node in op_nodes:
            gate_name = node.op.name.lower()
            if gate_name not in gate_type_map:
                continue  # 跳过不支持的门类型

            # 门类型独热编码
            gate_feat = torch.zeros(num_gate_types)
            gate_feat[gate_type_map[gate_name]] = 1.0

            # 量子比特掩码
            qubit_feat = torch.zeros(num_qubits)
            for q in node.qargs:
                qubit_feat[_get_qubit_index(q)] = 1.0

            # 合并特征
            node_feat = torch.cat([gate_feat, qubit_feat])
            node_features.append(node_feat)

        if not node_features:
            # 空电路处理
            return Data(x=torch.empty((0, num_gate_types + num_qubits)), edge_index=torch.empty((2, 0), dtype=torch.long))

        # 收集边信息 (有向边)
        edge_index = []
        node_indices = {node: i for i, node in enumerate(op_nodes)}

        for i, node in enumerate(op_nodes):
            for pred in dag.predecessors(node):
                if pred in node_indices:
                    j = node_indices[pred]
                    edge_index.append([j, i])  # 前驱 -> 后继

        # 转换为PyG格式
        x = torch.stack(node_features)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def _topology_to_data(self) -> Data:
        """将物理拓扑图转换为PyTorch Geometric的Data对象"""
        # 节点特征: 量子比特索引独热编码
        node_features = torch.eye(self.num_qubits)

        # 边索引: 从耦合图获取
        edges = self.coupling_map.get_edges()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def _generate_dataset(self) -> None:
        """生成并保存完整数据集"""
        print(f"生成{self.num_samples}个样本，拓扑类型: {self.topo_type}，量子比特数: {self.num_qubits}")
        print(f"使用的门类型: {self.basic_gates}")
        print(f"单量子比特门: {self.single_qubit_gates}")
        print(f"双量子比特门: {self.two_qubit_gates}")

        # 生成物理拓扑数据 (固定拓扑，所有样本共享)
        topo_data = self._topology_to_data()

        for i in trange(self.num_samples):
            try:
                # 1. 生成随机原始电路
                qc = self._generate_random_circuit()

                # 2. 经transpile得到优化后电路
                qco = transpile(
                    qc,
                    coupling_map=self.coupling_map,
                    optimization_level=2,
                    basis_gates=self.basic_gates + ['u']  # 使用指定的门集进行优化
                )

                # 3. 转换为DAG
                qc_dag = self.circuit_to_dag(qc)
                qco_dag = self.circuit_to_dag(qco)

                # 4. 转换为PyG Data对象
                g_data = self.dag_to_data(qc_dag, self.num_qubits)      # 优化前电路
                g_star_data = self.dag_to_data(qco_dag, self.num_qubits)  # 优化后电路

                # 5. 保存样本 (g, g*, t)
                sample = {
                    'g': g_data,       # 优化前的电路Data
                    'g_star': g_star_data,  # 优化后的电路Data
                    't': topo_data     # 物理拓扑Data
                }
            except Exception as e:
                print(f'生成失败：{e}')
            else:
                torch.save(sample, os.path.join(self.processed_dir, f"sample_{i}.pt"))

            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{self.num_samples} 个样本")

    def len(self) -> int:
        """返回数据集大小"""
        return self.num_samples

    def get(self, idx: int) -> Dict[str, Data]:
        """获取指定索引的样本"""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"样本索引 {idx} 超出范围")

        sample = torch.load(os.path.join(self.processed_dir, f"sample_{idx}.pt"))
        return sample


# -------------------------- 数据加载器使用示例 --------------------------
def get_qc_gcvae_dataloader(
    batch_size: int = 32,
    num_samples: int = 1000,
    num_qubits: int = 3,
    max_depth: int = 10,
    topo_type: str = "linear",
    basic_gates: List[str] = None,
    regenerate: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """创建QC-GCVAE模型的数据加载器，支持动态门类型配置"""
    dataset = QuantumCircuitDataset(
        root="./data",
        num_samples=num_samples,
        num_qubits=num_qubits,
        max_depth=max_depth,
        topo_type=topo_type,
        basic_gates=basic_gates,
        regenerate=regenerate
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=['g_x', 'g_star_x', 't_x']
    )


# 测试代码
if __name__ == "__main__":
    # 示例1: 使用默认门集 ['h', 'x', 'cx', 'swap']
    dataloader = get_qc_gcvae_dataloader(
        batch_size=8,
        num_samples=1000,
        num_qubits=20,
        max_depth=1000,
        topo_type="linear",
        regenerate=True
    )
    print(dataloader)
    
    # 测试加载一个批次
    for batch in dataloader:
        print("批次数据结构:")
        print(f"优化前电路: {batch['g']}")
        print(f"优化后电路: {batch['g_star']}")
        print(f"物理拓扑: {batch['t']}")

        # 打印第一个样本的基本信息
        first_g = batch['g'][0]
        first_g_star = batch['g_star'][0]
        first_t = batch['t'][0]

        print("\n第一个样本详情:")
        print(f"优化前电路 - 节点数: {first_g.x.size(0)}, 边数: {first_g.edge_index.size(1)}, 节点特征维度: {first_g.x.size(1)}")
        print(f"优化后电路 - 节点数: {first_g_star.x.size(0)}, 边数: {first_g_star.edge_index.size(1)}, 节点特征维度: {first_g_star.x.size(1)}")
        print(f"物理拓扑 - 节点数: {first_t.x.size(0)}, 边数: {first_t.edge_index.size(1)}")

        break  # 只测试一个批次
