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

def generate_jellyfish_network(N, d):
    """生成随机水母网络（修复节点度数超限问题）"""
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for node in range(N):
        # 当前节点已达最大度数，跳过
        if g.degree(node) >= d:
            continue
        # 可选邻居：不是自己、未连接且目标节点度数未达上限
        possible_neighbors = [
            n for n in range(N)
            if n != node
            and not g.has_edge(node, n)
            and g.degree(n) < d  # 新增：确保邻居节点度数不超标
        ]
        # 最多可添加的邻居数（不超过自身剩余额度）
        max_add = d - g.degree(node)
        if max_add <= 0 or not possible_neighbors:
            continue
        # 随机选择邻居并添加边
        num_to_add = min(max_add, len(possible_neighbors))
        neighbors = random.sample(possible_neighbors, num_to_add)
        g.add_edges_from([(node, n) for n in neighbors])
    return g

def generate_graph_by_model(graph_model: str, n: int, m_or_d: int = None, verbose=False):
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
            g = nx.grid_2d_graph(m=m_or_d, n=n)
            node_mapping = {node: idx for idx, node in enumerate(g.nodes)}
            return nx.relabel_nodes(g, node_mapping)
        if graph_model == 'random':
            # 随机拓扑：n个节点，平均度数m_or_d
            return generate_jellyfish_network(N=n, d=m_or_d)
        if graph_model == 'ring':
            # 环形拓扑：0-1-2-...-(n-1)-0（共n个节点，首尾相连）
            return nx.cycle_graph(n=n)  # 新增环形拓扑，使用networkx的cycle_graph

        raise ValueError(f'不支持的图模型: {graph_model}')

    g = generate()
    if verbose:
        print(f'生成 {graph_model} 图，节点数: {g.number_of_nodes()}')
        nx.draw(g)
    return g

def generate_coupling_map(graph_model: str, num_qubits: int):
    """生成指定类型的耦合图，支持ring拓扑"""
    assert num_qubits > 0, f"量子比特数必须为正数，实际为 {num_qubits}"

    def gen():
        if graph_model in ('linear', 'star', 'ring'):  # 新增ring支持
            # 线性/星型/环形拓扑：直接生成num_qubits个节点
            return generate_graph_by_model(graph_model, n=num_qubits)
        if graph_model == 'random':
            # 随机拓扑：指定节点数和度数
            return generate_graph_by_model(
                graph_model,
                n=num_qubits,
                m_or_d=random.randint(2, min(4, num_qubits-1))  # 限制度数范围
            )
        if graph_model == 'grid':
            # 网格拓扑：因子分解为最接近的两个数
            n, m = closest_factors(num_qubits)
            return generate_graph_by_model(graph_model, n=n, m_or_d=m)

        raise ValueError(f'不支持的拓扑类型: {graph_model}')

    g = gen()
    # 验证节点数是否与预期一致
    assert g.number_of_nodes() == num_qubits, \
        f"拓扑 {graph_model} 生成失败：预期 {num_qubits} 个节点，实际 {g.number_of_nodes()} 个"

    # 转换为Qiskit的耦合图（双向边）
    edges = list(g.edges) + [(v, u) for u, v in g.edges]  # 确保边是双向的
    return CouplingMap(edges)
