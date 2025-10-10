from torch_geometric.utils import to_networkx
import networkx as nx
import pygraphviz

def plot_pyg_data(data):
    g = to_networkx(data)
    layout = nx.layout.gra

