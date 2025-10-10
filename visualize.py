from torch_geometric.utils import to_networkx


def plot_pyg_data(data):
    g = to_networkx(data)
    

