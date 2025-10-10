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
from topologies import generate_coupling_map


def _get_qubit_index(q) -> int:
    """获取量子比特的索引，兼容不同Qiskit版本"""
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
    支持多拓扑、多优化电路的量子电路数据集
    使用字符串指定transpile的布局和路由方法，更简洁高效
    """
    def __init__(self,
                 root: str = "./data_multi_topo",
                 base_num_samples: int = 200,  # 基础原始电路数量
                 num_qubits: int = 4,
                 max_depth: int = 10,
                 topo_types: List[str] = None,  # 多拓扑类型列表
                 basic_gates: List[str] = None,
                 regenerate: bool = False):
        super().__init__(root)

        # 1. 基础参数初始化
        self.basic_gates = basic_gates if basic_gates is not None else ['h', 'x', 'cx', 'swap']
        self.base_num_samples = base_num_samples
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        # 支持的拓扑类型（默认：线性、星型、网格）
        self.topo_types = topo_types if topo_types is not None else ["linear", "star", "grid"]
        self._validate_topo_types()

        # 2. 门类型映射与拆分
        self.gate_class_map = self._create_gate_class_map()
        self.single_qubit_gates, self.two_qubit_gates = self._split_gate_types()

        # 3. 配置transpile参数组合（使用字符串指定方法）
        self.param_combinations = self._get_transpile_param_combinations()
        # 总样本数 = 原始电路数 × 拓扑数 × 参数组合数
        self.total_samples = base_num_samples * len(self.topo_types) * len(self.param_combinations)

        # 4. 目录与数据生成控制
        os.makedirs(self.processed_dir, exist_ok=True)
        if regenerate or not self._check_processed_exists():
            self._generate_dataset()

    def _validate_topo_types(self) -> None:
        """验证拓扑类型是否被支持"""
        supported_topo = ["linear", "star", "grid", "random"]  # 与generate_coupling_map对应
        for topo in self.topo_types:
            if topo not in supported_topo:
                raise ValueError(f"拓扑类型 {topo} 不支持，可选类型：{supported_topo}")

    def _create_gate_class_map(self) -> Dict[str, type]:
        """建立门名称到门类的映射"""
        supported_gates = {
            'h': HGate, 'x': XGate, 'z': ZGate, 't': TGate, 's': SGate,
            'cx': CXGate, 'swap': SwapGate
        }
        for gate_name in self.basic_gates:
            if gate_name not in supported_gates:
                raise ValueError(f"不支持的门类型: {gate_name}，支持类型：{list(supported_gates.keys())}")
        return {gate_name: supported_gates[gate_name] for gate_name in self.basic_gates}

    def _split_gate_types(self) -> Tuple[List[str], List[str]]:
        """区分单/双量子比特门"""
        two_qubit_gate_names = ['cx', 'swap']
        single, two = [], []
        for gate_name in self.basic_gates:
            if gate_name in two_qubit_gate_names:
                two.append(gate_name)
            else:
                single.append(gate_name)
        if not single:
            raise ValueError("至少需要指定一种单量子比特门")
        return single, two

    def _get_transpile_param_combinations(self) -> List[Dict[str, any]]:
        """
        定义transpile的参数组合（使用字符串指定方法）
        包含：布局方法（layout_method）、路由方法（routing_method）、优化级别（optimization_level）
        """
        # 1. 支持的布局方法（字符串形式）
        supported_layouts = [
            'trivial',    # 简单布局
            'dense',      # 稠密布局
            'sabre'       # Sabre布局（需耦合图，Qiskit内部自动处理）
        ]

        # 2. 支持的路由方法（字符串形式）
        supported_routers = [
            'basic',      # 基础交换路由
            'lookahead',  # 前瞻交换路由
            'sabre'       # Sabre路由（需耦合图，Qiskit内部自动处理）
        ]

        # 3. 优化级别（排除3，选择0/1/2）
        optimization_levels = [0, 1, 2]

        # 生成参数组合（筛选有效组合，避免冗余）
        param_combs = []
        for layout in supported_layouts:
            for router in supported_routers:
                # 对不同优化级别筛选：opt=0时减少组合，其他组合差别极小，造成冗余
                for opt_level in optimization_levels:
                    if opt_level == 0 and (layout != 'sabre' or router != 'sabre'):
                        continue  # opt=0时只保留Sabre组合
                    param_combs.append({
                        "layout_method": layout,
                        "routing_method": router,
                        "optimization_level": opt_level,
                        "param_id": len(param_combs)
                    })

        print(f"最终使用的transpile参数组合数：{len(param_combs)}")
        return param_combs

    def _check_processed_exists(self) -> bool:
        """检查所有样本是否已生成"""
        required_files = [
            f"sample_{base_idx}_topo{topo_idx}_param{param_idx}.pt"
            for base_idx in range(self.base_num_samples)
            for topo_idx in range(len(self.topo_types))
            for param_idx in range(len(self.param_combinations))
        ]
        return all(os.path.exists(os.path.join(self.processed_dir, f)) for f in required_files)

    def _get_coupling_map(self, topo_type: str) -> CouplingMap:
        """根据拓扑类型生成耦合图"""
        return generate_coupling_map(topo_type, self.num_qubits)

    def _generate_random_circuit(self) -> QuantumCircuit:
        """生成随机原始电路"""
        qc = QuantumCircuit(self.num_qubits)
        for _ in range(random.randint(1, self.max_depth)):
            use_two_qubit = (
                random.random() < 0.6 and
                self.num_qubits >= 2 and
                len(self.two_qubit_gates) > 0
            )
            if use_two_qubit:
                q1, q2 = random.sample(range(self.num_qubits), 2)
                gate_name = random.choice(self.two_qubit_gates)
                qc.append(self.gate_class_map[gate_name](), [q1, q2])
            else:
                q = random.choice(range(self.num_qubits))
                gate_name = random.choice(self.single_qubit_gates)
                qc.append(self.gate_class_map[gate_name](), [q])
        return qc

    @staticmethod
    def circuit_to_dag(qc: QuantumCircuit) -> DAGCircuit:
        """电路转DAG"""
        return circuit_to_dag(qc)

    def dag_to_data(self, dag: DAGCircuit, num_qubits: int) -> Data:
        """DAG转PyG Data对象"""
        gate_type_map = {gate_name: idx for idx, gate_name in enumerate(self.basic_gates)}
        num_gate_types = len(self.basic_gates)
        op_nodes = [node for node in dag.nodes() if isinstance(node, DAGOpNode)]

        # 节点特征
        node_features = []
        for node in op_nodes:
            gate_name = node.op.name.lower()
            if gate_name not in gate_type_map:
                continue
            # 门类型独热编码
            gate_feat = torch.zeros(num_gate_types)
            gate_feat[gate_type_map[gate_name]] = 1.0
            # 量子比特掩码
            qubit_feat = torch.zeros(num_qubits)
            for q in node.qargs:
                qubit_feat[_get_qubit_index(q)] = 1.0
            node_features.append(torch.cat([gate_feat, qubit_feat]))

        if not node_features:
            return Data(
                x=torch.empty((0, num_gate_types + num_qubits)),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )

        # 边索引
        edge_index = []
        node_indices = {node: i for i, node in enumerate(op_nodes)}
        for i, node in enumerate(op_nodes):
            for pred in dag.predecessors(node):
                if pred in node_indices:
                    edge_index.append([node_indices[pred], i])

        return Data(
            x=torch.stack(node_features),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

    def _topology_to_data(self, coupling_map: CouplingMap) -> Data:
        """拓扑耦合图转PyG Data对象"""
        node_features = torch.eye(self.num_qubits)  # 节点特征：比特索引独热编码
        edges = coupling_map.get_edges()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def _generate_dataset(self) -> None:
        """生成多拓扑、多优化电路的完整数据集"""
        print(f"=== 开始生成数据集 ===")
        print(f"量子比特数：{self.num_qubits}")
        print(f"最大深度：{self.max_depth}")
        print(f"基础原始电路数：{self.base_num_samples}")
        print(f"拓扑类型数：{len(self.topo_types)}（{self.topo_types}）")
        print(f"每个条件的优化样本数：{len(self.param_combinations)}")
        print(f"总样本数：{self.total_samples}")
        print(f"使用门类型：{self.basic_gates}")

        # 遍历：原始电路 → 拓扑 → 参数组合，生成所有样本
        sample_count = 0
        for base_idx in trange(self.base_num_samples, desc="原始电路进度"):
            # 生成1个原始电路（复用，对应多个拓扑和优化参数）
            qc = self._generate_random_circuit()
            qc_dag = self.circuit_to_dag(qc)
            g_data = self.dag_to_data(qc_dag, self.num_qubits)  # 优化前电路Data

            # 遍历所有拓扑
            for topo_idx, topo_type in enumerate(self.topo_types):
                coupling_map = self._get_coupling_map(topo_type)
                topo_data = self._topology_to_data(coupling_map)  # 拓扑Data

                # 遍历所有transpile参数组合，生成多个优化电路
                for param_idx, param in enumerate(self.param_combinations):
                    try:
                        # 调用transpile生成优化电路（使用字符串指定方法，Qiskit内部自动处理耦合图）
                        qco = transpile(
                            qc,
                            coupling_map=coupling_map,
                            layout_method=param["layout_method"],  # 字符串参数：如'sabre'
                            routing_method=param["routing_method"],  # 字符串参数：如'sabre'
                            optimization_level=param["optimization_level"],
                            basis_gates=self.basic_gates + ['u'],  # 兼容内部转换
                            seed_transpiler=base_idx + param_idx  # 固定种子确保复现
                        )

                        # 转换为DAG和PyG Data
                        qco_dag = self.circuit_to_dag(qco)
                        g_star_data = self.dag_to_data(qco_dag, self.num_qubits)  # 优化后电路Data

                        # 样本命名：区分原始电路、拓扑、参数组合
                        sample_filename = f"sample_{base_idx}_topo{topo_idx}_param{param_idx}.pt"
                        sample_path = os.path.join(self.processed_dir, sample_filename)

                        # 保存样本（包含条件信息和目标）
                        sample = {
                            "g": g_data,               # 条件1：优化前电路
                            "t": topo_data,            # 条件2：拓扑
                            "g_star": g_star_data,     # 目标：优化后电路
                            "meta": {                  # 元信息
                                "base_idx": base_idx,
                                "topo_type": topo_type,
                                "layout_method": param["layout_method"],
                                "routing_method": param["routing_method"],
                                "opt_level": param["optimization_level"]
                            }
                        }
                        torch.save(sample, sample_path)
                        sample_count += 1

                    except Exception as e:
                        print(f"\n生成样本失败（base{base_idx}_topo{topo_idx}_param{param_idx}）：{str(e)[:100]}")
                        raise e

        print(f"\n=== 生成完成 ===")
        print(f"成功生成样本数：{sample_count}/{self.total_samples}")

    def len(self) -> int:
        """返回总样本数"""
        return self.total_samples

    def get(self, idx: int) -> Dict[str, any]:
        """按索引获取样本（兼容DataLoader）"""
        param_per_topo = len(self.param_combinations)
        param_per_base = len(self.topo_types) * param_per_topo

        base_idx = idx // param_per_base
        remaining = idx % param_per_base
        topo_idx = remaining // param_per_topo
        param_idx = remaining % param_per_topo

        sample_filename = f"sample_{base_idx}_topo{topo_idx}_param{param_idx}.pt"
        sample_path = os.path.join(self.processed_dir, sample_filename)
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"样本 {sample_filename} 不存在，可能生成失败")
        return torch.load(sample_path)


# -------------------------- 数据加载器封装 --------------------------
def get_qc_gcvae_dataloader(
    batch_size: int = 32,
    base_num_samples: int = 200,
    num_qubits: int = 4,
    max_depth: int = 10,
    topo_types: List[str] = None,
    basic_gates: List[str] = None,
    regenerate: bool = False,
    shuffle: bool = True
) -> DataLoader:
    """创建支持多拓扑、多优化样本的Graph-CVAE数据加载器"""
    dataset = QuantumCircuitDataset(
        root="./data",
        base_num_samples=base_num_samples,
        num_qubits=num_qubits,
        max_depth=max_depth,
        topo_types=topo_types,
        basic_gates=basic_gates,
        regenerate=regenerate
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=['g_x', 'g_star_x', 't_x'],
        drop_last=True
    )


# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 测试配置：3种拓扑，包含Sabre方法的参数组合
    test_topo_types = ["linear", "star", "grid"]
    test_basic_gates = ['h', 'x', 'cx', 'swap', 'z']

    # 创建数据加载器（首次运行regenerate=True）
    dataloader = get_qc_gcvae_dataloader(
        batch_size=8,
        base_num_samples=10,  # 测试用小样本
        num_qubits=4,
        max_depth=10,
        topo_types=test_topo_types,
        basic_gates=test_basic_gates,
        regenerate=True
    )

    # 验证数据加载
    print(f"\n数据加载器批次数量：{len(dataloader)}")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n=== 批次 {batch_idx+1} 信息 ===")
        print(f"优化前电路（g）：{len(batch['g'])}个图")
        print(f"拓扑（t）：{len(batch['t'])}个图")
        print(f"优化后电路（g_star）：{len(batch['g_star'])}个图")

        # 打印首个样本的元信息（验证Sabre方法是否生效）
        first_meta = batch['meta'][0]
        print(f"首个样本参数：布局={first_meta['layout_method']}, 路由={first_meta['routing_method']}, 优化级别={first_meta['opt_level']}")

        if batch_idx >= 1:
            break  # 仅测试2个批次
