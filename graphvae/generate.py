import argparse
import networkx as nx
import matplotlib.pyplot as plt
import torch
import os
from mymodel import GraphVAE
from train import arg_parse  # Reuse argument parser


def generate_and_plot(args):
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)

    # Determine max_num_nodes from dataset (simplified)
    if args.dataset == 'enzymes':
        max_num_nodes = 128  # Typical for ENZYMES
    elif args.dataset == 'grid':
        max_num_nodes = 9  # From 3x3 grids
    else:
        max_num_nodes = args.max_num_nodes

    # Initialize model
    if args.feature_type == 'id':
        input_dim = max_num_nodes
    elif args.feature_type == 'deg':
        input_dim = 1
    else:
        input_dim = 2

    model = GraphVAE(input_dim, 64, 256, max_num_nodes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generate graphs
    generated_adjs = model.generate(num_samples=args.num_graphs)

    # Plot graphs
    plot_dir = 'generated_plots'
    os.makedirs(plot_dir, exist_ok=True)
    for i, adj in enumerate(generated_adjs):
        G = nx.from_numpy_array(adj)
        plt.figure(figsize=(6, 6))
        nx.draw(G, with_labels=True, node_color='lightblue')
        plot_path = os.path.join(plot_dir, f'generated_graph_{i}.png')
        plt.savefig(plot_path)
        print(f'Saved plot to {plot_path}')
    plt.close('all')


if __name__ == '__main__':
    parser = arg_parse()  # Reuse base parser
    # Add generation-specific arguments
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--num_graphs', type=int, default=5, help='Number of graphs to generate')
    args = parser.parse_args()
    generate_and_plot(args)
