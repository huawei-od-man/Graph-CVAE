import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

# 导入要测试的函数（请替换为实际模块名）
from circuit_info import (
    extract_circuit_info,
    compute_quantum_equivalence_data,
    compute_optimization_metrics
)


def test_extract_circuit_info():
    """测试电路信息提取函数，验证比特数、门数量、深度等指标"""
    # 创建包含多种门的测试电路
    qc = QuantumCircuit(2, 2)  # 2 Qb, 2 Cb
    qc.h(0)  # 单比特门
    qc.x(1)  # 单比特门
    qc.cx(0, 1)  # 双比特门
    qc.swap(0, 1)  # 双比特门
    qc.measure(0, 0)  # 测量门（应被排除）
    qc.measure(1, 1)

    # 提取信息
    info = extract_circuit_info(qc)

    # 验证结果
    assert info["num_qubits"] == 2, "比特数提取错误"
    assert info["total_gates"] == 4, "总门数错误（应排除测量门）"
    assert info["two_qubit_gates"] == 2, "双比特门计数错误"
    assert info["depth"] == 3, "电路深度计算错误（H→X并行，然后CX→SWAP串行）"


def test_compute_quantum_equivalence_data():
    """测试量子等价性数据计算，验证态矢量和酉矩阵正确性"""
    # 创建简单电路（H+CX生成Bell态）
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # 计算等价性数据
    result = compute_quantum_equivalence_data(qc)

    # 验证态矢量（Bell态 |Φ⁺⟩ = (|00⟩+|11⟩)/√2）
    expected_sv = Statevector.from_label("00").evolve(qc).data
    assert np.allclose(result["statevector"], expected_sv), "态矢量计算错误"

    # 验证酉矩阵（形状和内容）
    expected_unitary = Operator(qc).data
    assert result["unitary"].shape == (4, 4), "酉矩阵形状错误"
    assert np.allclose(result["unitary"], expected_unitary), "酉矩阵内容错误"


def test_compute_optimization_metrics():
    """测试优化指标计算，验证深度、门数比例的正确性"""
    # 原始电路
    qc_origin = QuantumCircuit(2)
    qc_origin.h(0)
    qc_origin.x(0)
    qc_origin.cx(0, 1)
    qc_origin.cx(0, 1)  # 原始：2单比特门，2双比特门，深度3

    # 优化后电路（假设减少了一半门）
    qc_opt = QuantumCircuit(2)
    qc_opt.x(0)  # 合并H+X为X（示例优化）
    qc_opt.cx(0, 1)  # 优化：1单比特门，1双比特门，深度2

    # 计算指标
    metrics = compute_optimization_metrics(qc_origin, qc_opt)

    # 验证结果
    assert metrics["depth_ratio"] == 0.5, "深度比例计算错误"
    assert metrics["total_gate_ratio"] == 0.5, "总门数比例错误"
    assert metrics["two_qubit_ratio"] == 0.5, "双比特门比例错误"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
