import networkx as nx
import numpy as np
import torch

# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes=20, max_num_nodes=1000, name='ENZYMES', node_attributes=True, graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: ' + str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/' + name + '/'
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path + name + '_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs


def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(min_num_nodes=10, name='DD', node_attributes=False, graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0:row * col], row=row, col=col, fname='figures/test')
    print('max num nodes', max_num_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features='id'):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            adj = nx.to_numpy_array(G)
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                np.asarray(adj) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == 'id':
                self.feature_all.append(np.identity(max_num_nodes))
            elif features == 'deg':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'struct':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, max_num_nodes - G.number_of_nodes()],
                                             'constant'),
                                      axis=1)
                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings,
                                                    [0, max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                self.feature_all.append(np.hstack([degs, clusterings]))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0

        adj_vectorized = adj_padded[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes))) == 1]
        # the following 2 lines recover the upper triangle of the adj matrix
        # recovered = np.zeros((self.max_num_nodes, self.max_num_nodes))
        # recovered[np.triu(np.ones((self.max_num_nodes, self.max_num_nodes)) ) == 1] = adj_vectorized
        # print(recovered)

        return {'adj': adj_padded,
                'adj_decoded': adj_vectorized,
                'features': self.feature_all[idx].copy()}
