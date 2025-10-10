import numpy as np
from typing import Dict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator, state_fidelity


# -------------------------- 辅助函数：电路基础信息提取 --------------------------
def extract_circuit_info(qc: QuantumCircuit) -> Dict[str, int]:
    """提取电路的比特数、门总数、深度、两比特门数"""
    # 排除测量门的统计
    non_measure_gates = [gate for gate in qc.data if gate.operation.name != 'measure']

    return {
        "num_qubits": qc.num_qubits,
        "total_gates": len(non_measure_gates),
        "depth": qc.depth(lambda g: g.operation.name in ['h', 'x', 'z', 't', 's', 'cx', 'swap']),
        "two_qubit_gates": len([g for g in non_measure_gates if g.operation.name in ['cx', 'swap']])
    }


# -------------------------- 辅助函数：电路等价性（statevector + unitary） --------------------------
def compute_quantum_equivalence_data(qc: QuantumCircuit) -> Dict[str, np.ndarray]:
    """计算电路的归一化statevector和unitary矩阵（自动移除测量门）"""
    # 复制电路并清除测量门，避免影响量子态计算
    qc_clean = qc.copy()
    qc_clean.remove_final_measurements()

    # 计算statevector（从|0>态初始化）
    sv = Statevector.from_instruction(qc_clean)
    # 计算unitary矩阵（仅支持无参数门电路）
    try:
        unitary = Operator(qc_clean).data
    except Exception as e:
        print(f"酉矩阵计算警告：{str(e)[:50]}，返回零矩阵")
        raise e

    return {
        "statevector": sv.data,  # 形状 [2^n]，已归一化
        "unitary": unitary       # 形状 [2^n, 2^n]
    }


# -------------------------- 辅助函数：优化后电路指标计算 --------------------------

def compute_optimization_metrics(qc_origin: QuantumCircuit, qc_opt: QuantumCircuit) -> Dict[str, float]:
    """计算优化电路指标（修复比特重排导致的保真度误差）

    暂时不计算保真度，因为涉及到量子比特重排的问题：qiskit的保真度必然是1，不需要计算。
    我们模型生成电路的保真度需要用类似匈牙利算法的办法把排列矩阵P先算出来，再用P作用一下，再计算保真度。

    """
    info_origin = extract_circuit_info(qc_origin)
    info_opt = extract_circuit_info(qc_opt)

    # 2. 计算比例类指标（保持不变）
    depth_ratio = round(
        info_opt["depth"] / info_origin["depth"] if info_origin["depth"] != 0 else 0, 4
    )
    total_gate_ratio = round(
        info_opt["total_gates"] / info_origin["total_gates"] if info_origin["total_gates"] != 0 else 0, 4
    )
    two_qubit_ratio = round(
        info_opt["two_qubit_gates"] / info_origin["two_qubit_gates"] if info_origin["two_qubit_gates"] != 0 else 0, 4
    )

    return {
        "depth_ratio": depth_ratio,
        "total_gate_ratio": total_gate_ratio,
        "two_qubit_ratio": two_qubit_ratio
    }
