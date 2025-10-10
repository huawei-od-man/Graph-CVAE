from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.transpiler import CouplingMap, Layout, TranspileLayout
from qiskit.circuit import Qubit
import numpy as np
from typing import List, Optional, Dict


# -------------------------- 修复后的比特重排函数 --------------------------
def reorder_statevector(sv: Statevector, permutation: List[int]) -> Statevector:
    """
    直接重排态矢量的比特顺序（正确逻辑）
    permutation[i] 表示新顺序中第i个比特对应原始顺序的第permutation[i]个比特
    例如：permutation=[1,0] 表示新比特0=原始比特1，新比特1=原始比特0
    """
    num_qubits = len(permutation)
    if int(np.log2(len(sv))) != num_qubits:
        raise ValueError("态矢量长度与比特数不匹配")

    # 生成新基的索引映射（核心修复）
    new_indices = []
    for idx in range(2**num_qubits):
        # 将索引转换为二进制（原始比特顺序）
        bits = np.binary_repr(idx, width=num_qubits)
        # 按permutation重排比特
        reordered_bits = ''.join([bits[num_qubits - 1 - permutation[i]] for i in range(num_qubits)])
        # 转换回整数索引
        new_idx = int(reordered_bits, 2)
        new_indices.append(new_idx)

    print(new_indices)

    # 按新索引重排态矢量数据
    reordered_data = sv.data[new_indices]
    return Statevector(reordered_data)


# -------------------------- 辅助函数：提取比特重排顺序 --------------------------
def get_bit_permutation(
    layout: Layout | TranspileLayout | None,
    original_qubits: List[Qubit]
) -> Optional[List[int]]:
    """从布局信息提取比特重排顺序"""
    if layout is None:
        return None

    target_layout: Optional[Layout] = None
    if isinstance(layout, TranspileLayout):
        target_layout = layout.initial_layout
    elif isinstance(layout, Layout):
        target_layout = layout
    else:
        print(f"警告：不支持的布局类型 {type(layout)}")
        return None

    if not isinstance(target_layout, Layout):
        print("警告：未获取到有效Layout对象")
        return None

    try:
        perm_phys = target_layout.to_permutation(original_qubits)
    except Exception as e:
        print(f"调用to_permutation失败：{str(e)[:50]}")
        return None

    if len(perm_phys) != len(original_qubits):
        print(f"警告：重排长度不匹配（{len(perm_phys)} vs {len(original_qubits)}）")
        return None

    final_layout: Optional[Layout] = layout.final_layout if isinstance(layout, TranspileLayout) else target_layout
    permutation = []

    for phys_idx in perm_phys:
        if isinstance(final_layout, Layout):
            try:
                if isinstance(phys_idx, Qubit):
                    phys_idx = phys_idx.index
                final_qubit = final_layout[phys_idx]
                final_idx = final_qubit.index if isinstance(final_qubit, Qubit) else final_qubit
                permutation.append(final_idx)
            except (KeyError, AttributeError):
                permutation.append(phys_idx)
        else:
            permutation.append(phys_idx)

    return permutation


# -------------------------- 测试用例：非对称态验证 --------------------------
def test_legacy_statevector_reorder_asymmetric():
    """使用非对称态（|01⟩）测试比特重排逻辑"""
    # 1. 创建原始电路（生成非对称态 |01⟩）
    qc_origin = QuantumCircuit(2)
    qc_origin.x(0)  # 仅翻转比特1
    print("=== 原始电路（生成非对称态 |01⟩）===")
    print(qc_origin.draw())

    # 原始态矢量
    sv_origin = Statevector.from_instruction(qc_origin)
    print(f"原始态矢量：{sv_origin.data.round(3)}（预期 [0, 1, 0, 0]）\n")

    # 2. 创建比特重排电路（模拟transpile的重排效果）
    qc_permuted = QuantumCircuit(2)
    qc_permuted.x(1)  # 比特交换后，原"翻转比特1"变为"翻转比特0"
    print("=== 比特重排后的电路（未纠正）===")
    print(qc_permuted.draw())

    # 重排后未纠正的态矢量
    sv_permuted = Statevector.from_instruction(qc_permuted)
    print(f"重排后态矢量（未纠正）：{sv_permuted.data.round(3)}（预期 [0, 0, 1, 0]）")

    # 未纠正的保真度
    fidelity_uncorrected = round(state_fidelity(sv_origin, sv_permuted), 4)
    print(f"未纠正的保真度：{fidelity_uncorrected}（预期 0.0）\n")

    # 3. 手动重排纠正
    permutation = [1, 0]  # 原始0→重排后1，原始1→重排后0
    inverse_perm = [permutation.index(i) for i in range(len(permutation))]
    sv_corrected = reorder_statevector(sv_permuted, inverse_perm)

    print("=== 比特重排纠正后 ===")
    print(f"重排后态矢量（纠正后）：{sv_corrected.data.round(3)}（预期 [0, 1, 0, 0]）")

    # 纠正后的保真度
    fidelity_corrected = round(state_fidelity(sv_origin, sv_corrected), 4)
    print(f"纠正后的保真度：{fidelity_corrected}（预期 1.0）\n")

    # 断言验证
    assert fidelity_uncorrected == 0.0, "未纠正时保真度应为0，测试失败"
    assert fidelity_corrected == 1.0, "纠正后保真度应为1，测试失败"
    print("✅ 非对称态测试通过：比特重排逻辑有效！")


# -------------------------- 测试用例：真实transpile场景验证 --------------------------
def test_transpile_real_permutation():
    """模拟真实transpile场景，验证端到端逻辑"""
    # 1. 原始电路（非对称态 |01⟩）
    qc_origin = QuantumCircuit(2)
    qc_origin.x(1)
    original_qubits = qc_origin.qubits

    # 2. 模拟硬件耦合图（仅支持1→0连接，触发布特重排）
    coupling_map = CouplingMap([[1, 0]])
    print("=== 真实transpile场景（耦合图触发重排）===")
    print(f"硬件耦合图：{coupling_map}")

    # 3. transpile优化（自动重排比特）
    qc_opt = transpile(
        qc_origin,
        coupling_map=coupling_map,
        layout_method="sabre",
        optimization_level=1
    )
    print("\nTranspile后的优化电路：")
    print(qc_opt.draw())

    # 4. 提取重排顺序并纠正态矢量
    sv_origin = Statevector.from_instruction(qc_origin)
    sv_opt_raw = Statevector.from_instruction(qc_opt.remove_final_measurements())

    permutation = get_bit_permutation(qc_opt.layout, original_qubits)
    print(f"\n提取的比特重排顺序（原始→优化后）：{permutation}（预期 [1, 0]）")

    # 纠正态矢量
    if permutation:
        inverse_perm = [permutation.index(i) for i in range(len(permutation))]
        sv_opt_corrected = reorder_statevector(sv_opt_raw, inverse_perm)
    else:
        sv_opt_corrected = sv_opt_raw

    # 5. 验证结果
    fidelity = round(state_fidelity(sv_origin, sv_opt_corrected), 4)
    print(f"\nTranspile后纠正的保真度：{fidelity}（预期 1.0）")

    assert fidelity == 1.0, "Transpile场景端到端测试失败"
    print("✅ Transpile真实场景测试通过！")


if __name__ == "__main__":
    test_legacy_statevector_reorder_asymmetric()
    print("-" * 50)
    test_transpile_real_permutation()
