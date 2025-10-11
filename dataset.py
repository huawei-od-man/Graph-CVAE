import os
import random
import qiskit
import qiskit.transpiler
import torch
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data, Dataset, Batch

from joblib import Parallel, delayed

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
ROUTING_METHOD_MAP = {"basic": 0, "sabre": 1,} #"lookahead": 2, }
# Lookahead routing is very slow. Bottlenect.

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
                 n_jobs: int = 8,
                 seed: int = 42):
        super().__init__(root)
        self.seed = seed
        self.n_jobs = n_jobs
        # 初始化参数与验证
        self.topo_types = topo_types if topo_types is not None else list(TOPO_TYPE_MAP.keys())
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

    # 修改_generate_all_samples方法
    def _generate_all_samples(self) -> None:
        """生成所有样本（含量子态、优化指标、元信息）"""
        print(f"\n=== 开始生成数据集 ===")
        print(f"基础样本数：{self.base_num_samples} | 拓扑类型：{self.topo_types} | 总样本数：{self.total_samples}")
        print(f"比特数：{self.num_qubits} | 门类型：{self.basic_gates} | 最大电路深度：{self.max_depth}")

        # 准备所有任务参数
        tasks = []
        for base_idx in trange(self.base_num_samples):
            # 先生成原始电路（单进程生成，避免随机种子冲突）
            qc_origin = self._generate_random_origin_circuit()
            origin_info = extract_circuit_info(qc_origin)
            origin_quantum = compute_quantum_equivalence_data(qc_origin)
            origin_dag = circuit_to_dag(qc_origin)
            g_data = self._dag_to_pyg_data(origin_dag)

            for topo_idx, topo_type in enumerate(self.topo_types):
                topo_code = TOPO_TYPE_MAP[topo_type]
                coupling_map = generate_coupling_map(topo_type, self.num_qubits)
                topo_data = Data(
                    x=torch.eye(self.num_qubits),
                    edge_index=torch.tensor(coupling_map.get_edges(), dtype=torch.long).t().contiguous()
                )

                for param_idx, param in enumerate(self.param_combinations):
                    tasks.append(dict(
                        base_idx=base_idx, topo_idx=topo_idx, param_idx=param_idx,
                        qc_origin=qc_origin, origin_info=origin_info, origin_quantum=origin_quantum,
                        g_data=g_data, topo_code=topo_code, coupling_map=coupling_map, topo_data=topo_data,
                        param=param,
                    ))

        # 多进程处理任务
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._process_single_sample)(**task) for task in tasks
        )

        sample_count = sum(results)
        print(f"\n=== 数据集生成完成 ===")
        print(f"成功生成样本数：{sample_count}/{self.total_samples}")
        if sample_count < self.total_samples:
            print(f"提示：部分样本生成失败，可检查日志定位问题（如门类型、拓扑参数）")

    def _process_single_sample(self, base_idx, topo_idx, param_idx,
                               qc_origin, origin_info, origin_quantum, g_data,
                               topo_code, coupling_map, topo_data,
                               param):
        """单样本处理函数（供多进程调用）"""
        try:
            qc_opt = transpile(
                qc_origin,
                coupling_map=coupling_map,
                layout_method=param["layout_method"],
                routing_method=param["routing_method"],
                optimization_level=param["optimization_level"],
                basis_gates=self.basic_gates,
                seed_transpiler=base_idx + topo_idx + param_idx + self.seed
            )

            if qc_opt.num_qubits != self.num_qubits:
                return 0

            opt_info = extract_circuit_info(qc_opt)
            opt_quantum = compute_quantum_equivalence_data(qc_opt)
            opt_metrics = compute_optimization_metrics(qc_origin, qc_opt)
            opt_dag = circuit_to_dag(qc_opt)
            g_star_data = self._dag_to_pyg_data(opt_dag)

            layout_code = LAYOUT_METHOD_MAP[param["layout_method"]]
            routing_code = ROUTING_METHOD_MAP[param["routing_method"]]

            sample = {
                "g": g_data,
                "t": topo_data,
                "g_star": g_star_data,
                "quantum_origin": {
                    "statevector": torch.tensor(origin_quantum["statevector"], dtype=torch.complex64),
                    "unitary": torch.tensor(origin_quantum["unitary"], dtype=torch.complex64)
                },
                "quantum_optimized": {
                    "statevector": torch.tensor(opt_quantum["statevector"], dtype=torch.complex64),
                    "unitary": torch.tensor(opt_quantum["unitary"], dtype=torch.complex64)
                },
                "optimization_metrics": opt_metrics,
                "circuit_origin_info": origin_info,
                "circuit_optimized_info": opt_info,
                "meta": {
                    "base_idx": base_idx,
                    "topo_type": topo_code,
                    "layout_method": layout_code,
                    "routing_method": routing_code,
                    "optimization_level": param["optimization_level"]
                }
            }

            sample_path = os.path.join(
                self.processed_dir,
                f"sample_b{base_idx}_t{topo_idx}_p{param_idx}.pt"
            )
            torch.save(sample, sample_path)
            return 1
        except qiskit.transpiler.exceptions.TranspilerError as e:
            print(f"\n样本生成失败（b{base_idx}_t{topo_idx}_p{param_idx}）：{e}")
            return 0

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
