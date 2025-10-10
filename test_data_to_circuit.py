from qiskit import QuantumCircuit
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import torch
from dataset import data_to_quantum_circuit
import pytest


# -------------------------- 测试函数 --------------------------
def test_data_to_circuit():
    # 1. 创建示例Data对象（模拟电路：H(0) → CX(0,1) → SWAP(1,2)）
    num_qubits = 3
    gate_types = ['h', 'x', 'cx', 'swap']  # 与编码时的门类型一致
    num_gate_types = len(gate_types)

    # 节点特征：门类型（独热）+ 比特掩码
    node0 = torch.zeros(num_gate_types + num_qubits)
    node0[gate_types.index('h')] = 1.0       # H门
    node0[num_gate_types + 0] = 1.0          # 作用于比特0

    node1 = torch.zeros(num_gate_types + num_qubits)
    node1[gate_types.index('cx')] = 1.0      # CX门
    node1[num_gate_types + 0] = 1.0          # 控制比特0
    node1[num_gate_types + 1] = 1.0          # 目标比特1

    node2 = torch.zeros(num_gate_types + num_qubits)
    node2[gate_types.index('swap')] = 1.0    # SWAP门
    node2[num_gate_types + 1] = 1.0          # 比特1
    node2[num_gate_types + 2] = 1.0          # 比特2

    # 边索引：0→1（H的输出是CX的输入），1→2（CX的输出是SWAP的输入）
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 有向边

    data = Data(x=torch.stack([node0, node1, node2]), edge_index=edge_index)

    # 2. 恢复电路
    recovered_qc = data_to_quantum_circuit(data, num_qubits, gate_types)

    # 3. 验证结果
    print("恢复的量子电路：")
    recovered_qc.draw()  # 可视化电路

    # 检查门的数量和类型
    assert len(recovered_qc.data) == 3, "门数量错误"
    assert recovered_qc.data[0][0].name == 'h', "第一个门应为H"
    assert recovered_qc.data[1][0].name == 'cx', "第二个门应为CX"
    assert recovered_qc.data[2][0].name == 'swap', "第三个门应为SWAP"
    print("测试通过：电路恢复正确")


if __name__ == "__main__":
    test_data_to_circuit()
