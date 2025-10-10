import pytest
from qiskit import QuantumCircuit
from dataset import compute_optimization_metrics


# -------------------------- 测试代码 --------------------------
def test_fidelity_with_permutation():
    # 1. 创建原始电路（2比特：H(0) → CX(0,1)）
    qc_origin = QuantumCircuit(2)
    qc_origin.h(0)
    qc_origin.cx(0, 1)
    print("原始电路：")
    print(qc_origin.draw())

    # 2. 模拟transpile后的比特重排（假设将逻辑比特0→1，1→0）
    # 实际中这由transpile自动完成，这里手动构造用于测试
    qc_opt = QuantumCircuit(2)
    qc_opt.h(1)  # 原H(0)变为H(1)（比特0和1交换）
    qc_opt.cx(1, 0)  # 原CX(0,1)变为CX(1,0)
    # 手动添加布局信息（模拟transpile的layout）
    from qiskit.transpiler.layout import Layout
    qc_opt.layout = Layout({0: 1, 1: 0})  # 逻辑比特0→物理比特1，逻辑比特1→物理比特0

    print("\n优化后电路（比特重排）：")
    print(qc_opt.draw())

    # 3. 计算指标（修复前会因比特顺序错误导致保真度0，修复后应为1）
    metrics = compute_optimization_metrics(qc_origin, qc_opt)
    print(f"\n计算结果：保真度={metrics['fidelity']}（预期1.0）")

    assert metrics["fidelity"] == 1.0, "比特重排导致保真度计算错误，修复失败"
    print("测试通过：比特重排已正确处理")

