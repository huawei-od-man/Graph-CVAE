import os
import random
import qiskit
import qiskit.transpiler
import torch
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data, Dataset, Batch

from converter import dag_to_pyg_data
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit

from tqdm import trange

from topologies import generate_coupling_map
from circuit_info import extract_circuit_info, compute_optimization_metrics, compute_quantum_equivalence_data
from utils import GATE_CLS_MAP


# -------------------------- 全局映射字典（字符串转整数） --------------------------
TOPO_TYPE_MAP = {"linear": 0, "star": 1, "grid": 2, "ring": 3, 'random': 4}
LAYOUT_METHOD_MAP = {"trivial": 0, "dense": 1, "sabre": 2}
ROUTING_METHOD_MAP = {"basic": 0, "lookahead": 1, "sabre": 2}


# -------------------------- 数据集类（含所有需求信息） --------------------------
class QuantumCircuitDataset(Dataset):
    def __init__(self,
                 root: str = "./data",
                 base_num_samples: int = 10,
                 num_qubits: int = 4,
                 max_depth: int = 5,
                 topo_types: List[str] = None,
                 basic_gates: List[str] = None,
                 regenerate: bool = False,
                 seed: int = 42):
        super().__init__(root)
        self.seed = seed
        # 初始化参数与验证
        self.topo_types = topo_types if topo_types is not None else ["linear", "star", "grid", "ring"]
        self._validate_topo_types()
        self.basic_gates = basic_gates if basic_gates is not None else list(GATE_CLS_MAP.keys())

        self.base_num_samples = base_num_samples
        self.num_qubits = num_qubits
        self.max_depth = max_depth

        # 门映射与transpile参数组合
        self.gate_class_map = self._build_gate_class_map()
        self.single_qubit_gates, self.two_qubit_gates = self._split_gate_types()
        self.param_combinations = self._generate_transpile_params()
        self.total_samples = base_num_samples * len(self.topo_types) * len(self.param_combinations)

        # 目录与数据生成触发
        os.makedirs(self.processed_dir, exist_ok=True)
        if regenerate or not self._check_samples_exist():
            self._generate_all_samples()

    def _validate_topo_types(self) -> None:
        """验证拓扑类型是否在支持列表中"""
        for topo in self.topo_types:
            if topo not in TOPO_TYPE_MAP:
                raise ValueError(f"不支持的拓扑：{topo}，支持列表：{list(TOPO_TYPE_MAP.keys())}")

    def _build_gate_class_map(self) -> Dict[str, type]:
        """建立门名称到Qiskit门类的映射"""
        for gate in self.basic_gates:
            if gate not in GATE_CLS_MAP:
                raise ValueError(f"不支持的门类型：{gate}，支持列表：{list(GATE_CLS_MAP.keys())}")

        return {g: GATE_CLS_MAP[g] for g in self.basic_gates}

    def _split_gate_types(self) -> Tuple[List[str], List[str]]:
        """区分单比特门和两比特门"""
        two_qubit = [g for g in self.basic_gates if g in ['cx', 'swap']]
        single_qubit = [g for g in self.basic_gates if g not in two_qubit]
        if not single_qubit:
            raise ValueError("数据集必须包含至少一种单比特门")
        return single_qubit, two_qubit

    def _generate_transpile_params(self) -> List[Dict[str, any]]:
        """生成transpile的（布局+路由+优化级别）参数组合"""
        layouts = list(LAYOUT_METHOD_MAP.keys())
        routers = list(ROUTING_METHOD_MAP.keys())
        opt_levels = [0, 1, 2]

        params = []
        for layout in layouts:
            for router in routers:
                for opt in opt_levels:
                    # opt=0时仅保留Sabre组合（避免冗余样本）
                    if opt == 0 and not (layout == 'sabre' and router == 'sabre'):
                        continue
                    params.append({
                        "layout_method": layout,
                        "routing_method": router,
                        "optimization_level": opt,
                        "param_id": len(params)
                    })
        print(f"Transpile参数组合数：{len(params)}")
        return params

    def _check_samples_exist(self) -> bool:
        """检查所有样本文件是否已生成"""
        required_files = [
            f"sample_b{base}_t{topo}_p{param}.pt"
            for base in range(self.base_num_samples)
            for topo in range(len(self.topo_types))
            for param in range(len(self.param_combinations))
        ]
        return all(os.path.exists(os.path.join(self.processed_dir, f)) for f in required_files)

    def _generate_random_origin_circuit(self) -> QuantumCircuit:
        """生成随机原始电路（无测量门，用于后续优化）"""
        qc = QuantumCircuit(self.num_qubits)
        for _ in range(random.randint(1, self.max_depth)):
            # 随机选择单/两比特门
            if random.random() < 0.6 and self.num_qubits >= 2 and self.two_qubit_gates:
                # 生成两比特门
                q1, q2 = random.sample(range(self.num_qubits), 2)
                gate_cls = self.gate_class_map[random.choice(self.two_qubit_gates)]
                qc.append(gate_cls(), [q1, q2])
            else:
                # 生成单比特门
                q = random.choice(range(self.num_qubits))
                gate_cls = self.gate_class_map[random.choice(self.single_qubit_gates)]
                qc.append(gate_cls(), [q])
        return qc

    def _dag_to_pyg_data(self, dag: DAGCircuit):
        return dag_to_pyg_data(dag, self.num_qubits, self.basic_gates)

    def _generate_all_samples(self) -> None:
        """生成所有样本（含量子态、优化指标、原信息）"""
        print(f"\n=== 开始生成数据集 ===")
        print(f"基础样本数：{self.base_num_samples} | 拓扑类型：{self.topo_types} | 总样本数：{self.total_samples}")
        print(f"比特数：{self.num_qubits} | 门类型：{self.basic_gates} | 最大电路深度：{self.max_depth}")

        sample_count = 0
        for base_idx in trange(self.base_num_samples, desc="原始电路生成进度"):
            # 1. 生成原始电路并计算基础数据
            qc_origin = self._generate_random_origin_circuit()
            origin_info = extract_circuit_info(qc_origin)
            origin_quantum = compute_quantum_equivalence_data(qc_origin)
            origin_dag = circuit_to_dag(qc_origin)
            g_data = self._dag_to_pyg_data(origin_dag)

            # 2. 遍历所有拓扑类型
            for topo_idx, topo_type in enumerate(self.topo_types):
                topo_code = TOPO_TYPE_MAP[topo_type]
                # 生成耦合图（已修复star拓扑，节点数=num_qubits）
                coupling_map = generate_coupling_map(topo_type, self.num_qubits)
                # 拓扑图转PyG Data（节点特征：比特索引独热编码）
                topo_data = Data(
                    x=torch.eye(self.num_qubits),  # 每个比特对应一个节点，特征为独热编码
                    edge_index=torch.tensor(coupling_map.get_edges(), dtype=torch.long).t().contiguous()
                )

                # 3. 遍历所有transpile参数组合
                for param_idx, param in enumerate(self.param_combinations):
                    try:
                        # 优化电路（移除initial_layout约束，依赖拓扑修复逻辑）
                        qc_opt = transpile(                            qc_origin,
                            coupling_map=coupling_map,
                            layout_method=param["layout_method"],
                            routing_method=param["routing_method"],
                            optimization_level=param["optimization_level"],
                            basis_gates=self.basic_gates,
                            seed_transpiler=base_idx + topo_idx + param_idx + self.seed # 固定随机种子确保可复现
                        )

                        # 验证优化后电路比特数（与原始一致，依赖拓扑生成逻辑）
                        if qc_opt.num_qubits != self.num_qubits:
                            print(f"警告：优化后比特数({qc_opt.num_qubits})≠原始({self.num_qubits})，跳过该样本")
                            continue

                        # 4. 计算优化后电路的核心数据
                        opt_info = extract_circuit_info(qc_opt)
                        opt_quantum = compute_quantum_equivalence_data(qc_opt)
                        opt_metrics = compute_optimization_metrics(qc_origin, qc_opt)
                        # 优化后电路转 PyG Data
                        opt_dag = circuit_to_dag(qc_opt)
                        g_star_data = self._dag_to_pyg_data(opt_dag)

                        # 5. 元信息整数编码（适配批量化）
                        layout_code = LAYOUT_METHOD_MAP[param["layout_method"]]
                        routing_code = ROUTING_METHOD_MAP[param["routing_method"]]

                        # 6. 构造完整样本（含所有需求信息）
                        sample = {
                            # 1. 图数据（CVAE 输入/目标）
                            "g": g_data,                  # 原始电路图（输入条件1）
                            "t": topo_data,               # 拓扑图（输入条件2）
                            "g_star": g_star_data,        # 优化后电路图（模型目标）

                            # 2. 电路等价性数据（验证功能一致性）
                            "quantum_origin": {
                                "statevector": torch.tensor(origin_quantum["statevector"], dtype=torch.complex64),
                                "unitary": torch.tensor(origin_quantum["unitary"], dtype=torch.complex64)
                            },
                            "quantum_optimized": {
                                "statevector": torch.tensor(opt_quantum["statevector"], dtype=torch.complex64),
                                "unitary": torch.tensor(opt_quantum["unitary"], dtype=torch.complex64)
                            },

                            # 3. 优化后电路指标（评估优化效果）
                            "optimization_metrics": opt_metrics,

                            # 4. 输入电路原信息（训练辅助特征/评估对比）
                            "circuit_origin_info": origin_info,
                            "circuit_optimized_info": opt_info,

                            # 5. 元信息（样本追溯/条件筛选）
                            "meta": {
                                "base_idx": base_idx,
                                "topo_type": topo_code,
                                "layout_method": layout_code,
                                "routing_method": routing_code,
                                "optimization_level": param["optimization_level"]
                            }
                        }

                        # 7. 保存样本
                        sample_path = os.path.join(
                            self.processed_dir,
                            f"sample_b{base_idx}_t{topo_idx}_p{param_idx}.pt"
                        )
                        torch.save(sample, sample_path)
                        sample_count += 1

                    except qiskit.transpiler.exceptions.TranspilerError as e:
                        print(f"\n样本生成失败（b{base_idx}_t{topo_idx}_p{param_idx}）：{str(e)[:80]}")
                        continue

        print(f"\n=== 数据集生成完成 ===")
        print(f"成功生成样本数：{sample_count}/{self.total_samples}")
        if sample_count < self.total_samples:
            print(f"提示：部分样本生成失败，可检查日志定位问题（如门类型、拓扑参数）")

    def len(self) -> int:
        return self.total_samples

    def get(self, idx: int) -> Dict[str, any]:
        """按索引获取样本（Dataset 抽象方法实现，适配 DataLoader）"""
        # 计算索引对应的 base/topo/param 编号
        param_per_topo = len(self.param_combinations)
        param_per_base = len(self.topo_types) * param_per_topo

        base_idx = idx // param_per_base
        remaining = idx % param_per_base
        topo_idx = remaining // param_per_topo
        param_idx = remaining % param_per_topo

        # 读取样本
        sample_path = os.path.join(
            self.processed_dir,
            f"sample_b{base_idx}_t{topo_idx}_p{param_idx}.pt"
        )
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"样本文件不存在：{sample_path}（可能生成失败）")
        return torch.load(sample_path)


# -------------------------- 自定义批处理函数（适配复杂样本结构） --------------------------
def custom_collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """将多个样本合并为批次，处理图、量子态、指标等不同类型数据"""
    batched_data = {}

    # 1. 处理 PyG 图数据（用 Batch.from_data_list 合并）
    graph_keys = ["g", "t", "g_star"]
    for key in graph_keys:
        batched_data[key] = Batch.from_data_list([sample[key] for sample in batch])

    # 2. 处理量子态数据（复数张量堆叠，形状 [batch_size, ...]）
    batched_data["quantum_origin"] = {
        "statevector": torch.stack([s["quantum_origin"]["statevector"] for s in batch]),
        "unitary": torch.stack([s["quantum_origin"]["unitary"] for s in batch])
    }
    batched_data["quantum_optimized"] = {
        "statevector": torch.stack([s["quantum_optimized"]["statevector"] for s in batch]),
        "unitary": torch.stack([s["quantum_optimized"]["unitary"] for s in batch])
    }

    # 3. 处理优化指标（浮点数转张量，形状 [batch_size]）
    batched_data["optimization_metrics"] = {
        # "fidelity": torch.tensor([s["optimization_metrics"]["fidelity"] for s in batch], dtype=torch.float32),
        "depth_ratio": torch.tensor([s["optimization_metrics"]["depth_ratio"] for s in batch], dtype=torch.float32),
        "total_gate_ratio": torch.tensor([s["optimization_metrics"]["total_gate_ratio"] for s in batch], dtype=torch.float32),
        "two_qubit_ratio": torch.tensor([s["optimization_metrics"]["two_qubit_ratio"] for s in batch], dtype=torch.float32)
    }

    # 4. 处理电路原信息（整数转张量，形状 [batch_size]）
    batched_data["circuit_origin_info"] = {
        "num_qubits": torch.tensor([s["circuit_origin_info"]["num_qubits"] for s in batch], dtype=torch.long),
        "total_gates": torch.tensor([s["circuit_origin_info"]["total_gates"] for s in batch], dtype=torch.long),
        "depth": torch.tensor([s["circuit_origin_info"]["depth"] for s in batch], dtype=torch.long),
        "two_qubit_gates": torch.tensor([s["circuit_origin_info"]["two_qubit_gates"] for s in batch], dtype=torch.long)
    }
    batched_data["circuit_optimized_info"] = {
        "num_qubits": torch.tensor([s["circuit_optimized_info"]["num_qubits"] for s in batch], dtype=torch.long),
        "total_gates": torch.tensor([s["circuit_optimized_info"]["total_gates"] for s in batch], dtype=torch.long),
        "depth": torch.tensor([s["circuit_optimized_info"]["depth"] for s in batch], dtype=torch.long),
        "two_qubit_gates": torch.tensor([s["circuit_optimized_info"]["two_qubit_gates"] for s in batch], dtype=torch.long)
    }

    # 5. 处理元信息（整数张量，形状 [batch_size]）
    batched_data["meta"] = {
        "base_idx": torch.tensor([s["meta"]["base_idx"] for s in batch], dtype=torch.long),
        "topo_type": torch.tensor([s["meta"]["topo_type"] for s in batch], dtype=torch.long),
        "layout_method": torch.tensor([s["meta"]["layout_method"] for s in batch], dtype=torch.long),
        "routing_method": torch.tensor([s["meta"]["routing_method"] for s in batch], dtype=torch.long),
        "optimization_level": torch.tensor([s["meta"]["optimization_level"] for s in batch], dtype=torch.long)
    }

    return batched_data


# -------------------------- 数据加载器封装（一键获取训练/评估数据） --------------------------
def get_dataloader(
    batch_size: int = 16,
    base_num_samples: int = 100,
    num_qubits: int = 4,
    max_depth: int = 10,
    topo_types: List[str] = None,
    basic_gates: List[str] = None,
    regenerate: bool = False,
    shuffle: bool = True
) -> 'DataLoader':
    """创建 QC Graph-CVAE 的数据加载器，直接用于训练/评估"""
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
        collate_fn=custom_collate_fn,  # 关键：用自定义批处理适配复杂样本
        drop_last=True,  # 丢弃最后一个不完整批次
        pin_memory=True  # 加速 GPU 数据传输（如需 CPU 训练可设为 False）
    )
