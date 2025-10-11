import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from torch_geometric.data import Data
from qiskit.converters import circuit_to_dag

# 导入待测试函数（替换为实际模块名）
from converter import dag_to_pyg_data, data_to_quantum_circuit
from utils import get_qubit_index


# -------------------------- 通用测试配置 --------------------------
@pytest.fixture(scope="module")
def basic_config():
    """通用测试配置：基础门列表、电路比特数、示例电路"""
    basic_gates = ["h", "x", "cx", "swap"]  # 常见门类型
    num_qubits = 2  # 2比特电路（常见场景）

    # 示例电路：含单比特门（H/X）、双比特门（CX/SWAP），带依赖关系
    qc = QuantumCircuit(num_qubits)
    qc.h(0)  # 门1：H(0)
    qc.x(1)  # 门2：X(1)
    qc.cx(0, 1)  # 门3：CX(0→1)（依赖门1、2）
    qc.swap(0, 1)  # 门4：SWAP(0,1)（依赖门3）

    return {"basic_gates": basic_gates, "num_qubits": num_qubits, "qc": qc}


# -------------------------- 测试 dag_to_pyg_data --------------------------
def test_dag_to_pyg_data_common(basic_config):
    """测试常见场景：含多类型门的电路→PyG Data"""
    basic_gates = basic_config["basic_gates"]
    num_qubits = basic_config["num_qubits"]
    qc = basic_config["qc"]

    # 1. 电路转DAG，再转Data
    dag = circuit_to_dag(qc)
    data = dag_to_pyg_data(dag, num_qubits=num_qubits, basic_gates=basic_gates)

    # 2. 验证核心属性（节点数、特征长度、边数）
    assert isinstance(data, Data), "返回对象不是PyG Data"
    assert data.x.shape == (4, len(basic_gates) + num_qubits), \
        f"节点特征形状错误，预期(4,6)，实际{data.x.shape}"  # 4个门，特征长度=4+2=6

    # 3. 验证关键节点特征（H门、CX门）
    # H门（第0个节点）：门类型独热[1,0,0,0] + 比特掩码[1,0]
    assert torch.allclose(data.x[0], torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)), "H门特征错误"
    # CX门（第2个节点）：门类型独热[0,0,1,0] + 比特掩码[1,1]
    assert torch.allclose(data.x[2], torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.float)), "CX门特征错误"

    # 4. 验证依赖边（至少包含 H→CX、X→CX、CX→SWAP 的边）
    assert data.edge_index.shape[1] >= 3, f"边数不足，预期≥3，实际{data.edge_index.shape[1]}"

    # 5. 验证edge_index的合法性：所有结点都在范围内。
    assert all(0 <= v < data.num_nodes for v in data.edge_index.reshape(-1))


# -------------------------- 测试 data_to_quantum_circuit --------------------------
def test_data_to_quantum_circuit_common(basic_config):
    """测试常见场景：PyG Data→含多类型门的电路"""
    basic_gates = basic_config["basic_gates"]
    num_qubits = basic_config["num_qubits"]
    qc_origin = basic_config["qc"]

    # 1. 先转Data（复用前序转换结果）
    dag = circuit_to_dag(qc_origin)
    data = dag_to_pyg_data(dag, num_qubits=num_qubits, basic_gates=basic_gates)

    # 2. Data转电路
    qc_restored = data_to_quantum_circuit(
        data=data, num_qubits=num_qubits, gate_types=basic_gates
    )

    # 3. 验证核心属性（比特数、门数量、门类型顺序）
    assert qc_restored.num_qubits == num_qubits, "电路比特数错误"
    assert len(qc_restored.data) == 4, f"门数量错误，预期4，实际{len(qc_restored.data)}"

    # 4. 验证门类型和顺序与原始一致
    orig_gate_names = [gate[0].name for gate in qc_origin.data]
    restored_gate_names = [gate[0].name for gate in qc_restored.data]
    assert restored_gate_names == orig_gate_names, "门类型/顺序错误"

    # 5. 验证关键门的作用比特
    # CX门（第2个门）：作用于(0,1)
    assert [get_qubit_index(qb) for qb in qc_restored.data[2][1]] == [0, 1], "CX门比特错误"
    # SWAP门（第3个门）：作用于(0,1)
    assert [get_qubit_index(qb) for qb in qc_restored.data[3][1]] == [0, 1], "SWAP门比特错误"


def test_dag_to_pyg_data_empty_circuit():
    """测试空电路转换，确保节点和边为空"""
    qc = QuantumCircuit(2)  # 空电路
    dag = circuit_to_dag(qc)
    data = dag_to_pyg_data(dag, 2, ['h', 'cx'])

    assert data.num_nodes == 0, "空电路应包含0个节点"
    assert data.edge_index.numel() == 0, "空电路应包含0条边"


def test_dag_to_pyg_data_with_unsupported_gate():
    """测试遇到不支持的门时是否直接报错"""
    # 创建包含不支持门（rz）的电路
    qc = QuantumCircuit(2)
    qc.h(0)  # 支持的门
    qc.rz(0.5, 0)  # 不支持的门
    qc.cx(0, 1)  # 支持的门
    dag = circuit_to_dag(qc)

    # 定义仅支持h、cx、s的基础门列表
    basic_gates = ['h', 'cx', 's']

    # 预期会抛出ValueError，提示不支持rz门
    with pytest.raises(ValueError) as excinfo:
        dag_to_pyg_data(dag, num_qubits=2, basic_gates=basic_gates)

    # 验证错误信息是否包含关键内容
    assert "rz" in str(excinfo.value)


def test_dag_to_pyg_data_all_supported_gates():
    """测试所有门都支持的情况，确保正常转换"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.s(1)
    dag = circuit_to_dag(qc)

    basic_gates = ['h', 'cx', 's']
    data = dag_to_pyg_data(dag, 2, basic_gates)

    # 验证节点数量（3个支持的门）
    assert data.num_nodes == 3
    # 验证边索引不为空（存在依赖关系）
    assert data.edge_index.numel() > 0


def test_converter_stressed(tmp_path):
    from dataset import QuantumCircuitDataset

    dataset = QuantumCircuitDataset(
        tmp_path,
        base_num_samples=1,
        num_qubits=10,
        max_depth=100,
        regenerate=True,
        topo_types=['grid']
    )
    for i in range(dataset.total_samples):
        sample = dataset.get(i)
        data_to_quantum_circuit(sample['g'], dataset.num_qubits, dataset.basic_gates)
        data_to_quantum_circuit(sample['g_star'], dataset.num_qubits, dataset.basic_gates)
        # Should not raise any exception.
