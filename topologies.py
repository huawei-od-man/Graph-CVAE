import networkx as nx
import random
from qiskit.transpiler import CouplingMap

def closest_factors(num):
    """找到最接近的两个因子，用于网格拓扑"""
    factors = []
    for i in range(1, int(num**0.5) + 1):
        if num % i == 0:
            factors.append((i, num // i))
    return factors[-1] if factors else (1, num)


def create_jellyfish_graph(n, tentacle_lengths=None):
    """
    创建水母图

    参数:
    n: 总节点数
    tentacle_lengths: 触手长度列表，如果不指定则自动分配
    """
    G = nx.Graph()

    # 添加中心节点
    G.add_node(0)

    # 计算可用节点数（减去中心节点）
    available_nodes = n - 1

    # 如果没有指定触手长度，自动分配
    if tentacle_lengths is None:
        # 尽量创建多个触手，每个触手长度适中
        num_tentacles = min(available_nodes, max(2, available_nodes // 3))
        base_length = available_nodes // num_tentacles
        remainder = available_nodes % num_tentacles

        tentacle_lengths = [base_length] * num_tentacles
        for i in range(remainder):
            tentacle_lengths[i] += 1
    else:
        # 确保触手长度总和不超过可用节点数
        if sum(tentacle_lengths) > available_nodes:
            raise ValueError("触手长度总和超过了可用节点数")

    # 构建触手
    current_node = 1
    for i, length in enumerate(tentacle_lengths):
        if length > 0:
            # 连接中心节点到触手的第一个节点
            G.add_edge(0, current_node)

            # 构建触手的链式结构
            for j in range(1, length):
                G.add_edge(current_node + j - 1, current_node + j)

            current_node += length

    return G


def generate_graph_by_model(graph_model: str, n: int):
    """生成指定类型的图，新增ring拓扑支持"""
    def generate():
        if graph_model == 'linear':
            # 线性拓扑：0-1-2-...-(n-1)
            return nx.path_graph(n=n)
        if graph_model == 'star':
            # 星型拓扑：中心0连接1,2,...,n-1（共n个节点）
            return nx.star_graph(n-1)
        if graph_model == 'grid':
            # 网格拓扑：m行n列，总节点数m*n
            n_, m = closest_factors(n)
            g = nx.grid_2d_graph(m=m, n=n_)
            node_mapping = {node: idx for idx, node in enumerate(g.nodes)}
            return nx.relabel_nodes(g, node_mapping)
        if graph_model == 'random':
            # 随机拓扑：n个节点，平均度数m_or_d
            return create_jellyfish_graph(n)
        if graph_model == 'ring':
            # 环形拓扑：0-1-2-...-(n-1)-0（共n个节点，首尾相连）
            return nx.cycle_graph(n=n)  # 新增环形拓扑，使用networkx的cycle_graph

        raise ValueError(f'不支持的图模型: {graph_model}')

    g = generate()
    assert nx.is_connected(g), "Must be connected graph!!"

    return g

def generate_coupling_map(graph_model: str, num_qubits: int):
    """生成指定类型的耦合图，支持ring拓扑"""
    assert num_qubits > 0, f"量子比特数必须为正数，实际为 {num_qubits}"

    g = generate_graph_by_model(graph_model, num_qubits)
    # 验证节点数是否与预期一致
    assert g.number_of_nodes() == num_qubits, \
        f"拓扑 {graph_model} 生成失败：预期 {num_qubits} 个节点，实际 {g.number_of_nodes()} 个"

    # 转换为Qiskit的耦合图（双向边）
    edges = list(g.edges) + [(v, u) for u, v in g.edges]  # 确保边是双向的
    return CouplingMap(edges)
