import pytest
import networkx as nx
from qiskit.transpiler import CouplingMap
from topologies import closest_factors, create_jellyfish_graph, generate_graph_by_model, generate_coupling_map

def test_closest_factors():
    # 测试完全平方数
    assert closest_factors(16) == (4, 4)
    # 测试质数
    assert closest_factors(7) == (1, 7)
    # 测试普通合数
    assert closest_factors(12) == (3, 4)
    # 测试1
    assert closest_factors(1) == (1, 1)

def test_generate_jellyfish_network():
    N = 5
    g = create_jellyfish_graph(N)
    # 验证节点数
    assert g.number_of_nodes() == N
    # 验证每个节点的度数不超过d
    # 验证无自环
    assert not any(g.has_edge(node, node) for node in g.nodes)
    assert nx.is_connected(g)

def test_generate_graph_by_model():
    # 测试线性拓扑
    linear = generate_graph_by_model('linear', 5)
    assert isinstance(linear, nx.Graph)
    assert linear.number_of_nodes() == 5
    assert linear.edges == nx.path_graph(5).edges

    # 测试星型拓扑
    star = generate_graph_by_model('star', 5)
    assert star.number_of_nodes() == 5
    assert star.edges == nx.star_graph(4).edges  # star_graph(n)生成n+1个节点

    # 测试网格拓扑
    grid = generate_graph_by_model('grid', 6)  # 2行3列
    assert grid.number_of_nodes() == 6
    assert isinstance(grid, nx.Graph)

    # 测试环形拓扑
    ring = generate_graph_by_model('ring', 5)
    assert ring.number_of_nodes() == 5
    assert ring.edges == nx.cycle_graph(5).edges

    # 测试随机拓扑
    random_graph = generate_graph_by_model('random', 5, 2)
    assert random_graph.number_of_nodes() == 5

def test_generate_coupling_map():
    # 测试线性耦合图
    linear_cm = generate_coupling_map('linear', 5)
    assert isinstance(linear_cm, CouplingMap)
    assert linear_cm.graph.num_edges() == 8  # 4条边双向展示

    # 测试环形耦合图
    ring_cm = generate_coupling_map('ring', 5)
    assert isinstance(ring_cm, CouplingMap)
    assert ring_cm.graph.num_edges() == 10  # 5条边双向展示

    # 测试网格耦合图（6个节点：2x3）
    grid_cm = generate_coupling_map('grid', 6)
    assert isinstance(grid_cm, CouplingMap)
    assert grid_cm.size() == 6

    # 测试随机耦合图
    random_cm = generate_coupling_map('random', 5)
    assert isinstance(random_cm, CouplingMap)
    assert random_cm.size() == 5
