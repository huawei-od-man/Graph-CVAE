import os
import tempfile
import torch
import pytest
from qiskit import QuantumCircuit
from torch_geometric.data import Data, Batch
from dataset import (
    QuantumCircuitDataset,
    custom_collate_fn,
    get_dataloader,
    TOPO_TYPE_MAP,
    LAYOUT_METHOD_MAP,
    ROUTING_METHOD_MAP
)


# 测试辅助函数：创建临时目录
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# 测试 QuantumCircuitDataset 初始化
def test_dataset_initialization(temp_dir):
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        base_num_samples=10,
        num_qubits=2,
        max_depth=5,
        topo_types=["linear"],
        basic_gates=['h', 'x', 'cx'],
        regenerate=True,
        seed=42
    )

    # 验证基本属性
    assert dataset.base_num_samples == 10
    assert dataset.num_qubits == 2
    assert dataset.max_depth == 5
    assert dataset.topo_types == ["linear"]
    assert dataset.basic_gates == ['h', 'x', 'cx']
    assert os.path.exists(dataset.processed_dir)


# 测试 _validate_topo_types 方法
def test_validate_topo_types(temp_dir):
    # 测试合法拓扑
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        topo_types=["linear", "star"],
        regenerate=False
    )
    dataset._validate_topo_types()  # 不应抛出异常

    # 测试非法拓扑
    with pytest.raises(ValueError):
        dataset = QuantumCircuitDataset(
            root=temp_dir,
            topo_types=["invalid_topo"],
            regenerate=False
        )
        dataset._validate_topo_types()


# 测试 _build_gate_class_map 方法
def test_build_gate_class_map(temp_dir):
    # 测试合法门类型
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        basic_gates=['h', 'x', 'cx'],
        regenerate=False
    )
    gate_map = dataset._build_gate_class_map()
    assert all(gate in gate_map for gate in ['h', 'x', 'cx'])

    # 测试非法门类型
    with pytest.raises(ValueError):
        dataset = QuantumCircuitDataset(
            root=temp_dir,
            basic_gates=['invalid_gate'],
            regenerate=False
        )
        dataset._build_gate_class_map()


# 测试 _split_gate_types 方法
def test_split_gate_types(temp_dir):
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        basic_gates=['h', 'x', 'cx', 'swap'],
        regenerate=False
    )
    single, two = dataset._split_gate_types()
    assert set(single) == {'h', 'x'}
    assert set(two) == {'cx', 'swap'}

    # 测试没有单比特门的情况
    with pytest.raises(ValueError):
        dataset = QuantumCircuitDataset(
            root=temp_dir,
            basic_gates=['cx', 'swap'],
            regenerate=False
        )
        dataset._split_gate_types()


# 测试 _generate_transpile_params 方法
def test_generate_transpile_params(temp_dir):
    dataset = QuantumCircuitDataset(root=temp_dir, regenerate=False)
    params = dataset._generate_transpile_params()

    # 验证参数组合数量（opt=0时仅保留sabre组合）
    assert len(params) == 1 + 3 * 3 * 2  # 1个opt=0组合 + 3*3*2个opt>0组合

    # 验证opt=0时的参数
    opt0_params = [p for p in params if p["optimization_level"] == 0]
    assert len(opt0_params) == 1
    assert opt0_params[0]["layout_method"] == "sabre"
    assert opt0_params[0]["routing_method"] == "sabre"


# 测试 _generate_random_origin_circuit 方法
def test_generate_random_origin_circuit(temp_dir):
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        num_qubits=2,
        max_depth=5,
        basic_gates=['h', 'x', 'cx'],
        regenerate=False
    )
    qc = dataset._generate_random_origin_circuit()

    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 2
    assert 1 <= qc.depth() <= 5  # 深度在1到max_depth之间


# 测试 get 方法和样本生成
def test_get_method(temp_dir):
    # 生成一个小数据集
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        base_num_samples=1,
        num_qubits=2,
        max_depth=3,
        topo_types=["linear"],
        basic_gates=['h', 'x', 'cx'],
        regenerate=True,
        seed=42
    )

    # 验证样本数量
    assert dataset.len() == len(dataset.param_combinations)

    # 验证get方法
    sample = dataset.get(0)
    assert isinstance(sample, dict)
    assert all(key in sample for key in ["g", "t", "g_star", "quantum_origin", "meta"])
    assert isinstance(sample["g"], Data)  # 原始电路图
    assert isinstance(sample["t"], Data)  # 拓扑图
    assert isinstance(sample["meta"], dict)


# 测试 custom_collate_fn 函数
def test_custom_collate_fn(temp_dir):
    # 生成两个样本
    dataset = QuantumCircuitDataset(
        root=temp_dir,
        base_num_samples=1,
        num_qubits=2,
        max_depth=3,
        topo_types=["linear"],
        basic_gates=['h', 'x', 'cx'],
        regenerate=True,
        seed=42
    )
    sample1 = dataset.get(0)
    sample2 = dataset.get(1) if dataset.len() > 1 else sample1

    # 批处理测试
    batch = custom_collate_fn([sample1, sample2])

    # 验证图数据批处理
    assert isinstance(batch["g"], Batch)
    assert batch["g"].num_graphs == 2

    # 验证量子态数据形状
    assert batch["quantum_origin"]["statevector"].shape[0] == 2
    assert batch["quantum_origin"]["unitary"].shape[0] == 2

    # 验证指标数据形状
    assert batch["optimization_metrics"]["depth_ratio"].shape == (2,)


# 测试 get_dataloader 函数
def test_get_dataloader(temp_dir):
    dataloader = get_dataloader(
        batch_size=2,
        base_num_samples=1,
        num_qubits=2,
        max_depth=3,
        topo_types=["linear"],
        basic_gates=['h', 'x', 'cx'],
        regenerate=True,
        shuffle=False
    )

    # 验证数据加载器
    assert len(dataloader) >= 1
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert batch["g"].num_graphs == 2  # 批大小为2
