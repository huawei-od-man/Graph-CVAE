import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import data
from mymodel import GraphVAE
from data import GraphAdjSampler

device = 0 if torch.cuda.is_available() else 'cpu'

LR_milestones = [500, 1000]


# Add to train.py (after train function)
def is_valid_graph(adj):
    """Check if adjacency matrix is symmetric and has no self-loops."""
    # No self-loops (diagonal must be 0)
    if not np.allclose(np.diag(adj), 0):
        return False
    # Symmetric
    if not np.allclose(adj, adj.T):
        return False
    return True


def evaluate(model, num_samples=10):
    """Evaluate generation quality: validity rate."""
    generated = model.generate(num_samples)
    valid_count = sum(1 for adj in generated if is_valid_graph(adj))
    return valid_count / num_samples if num_samples > 0 else 0.0


# Modify train function to include checkpointing and evaluation
def train(args, dataloader, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=0.5)  # Fixed gamma
    best_validity = 0.0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            g = data['g']
            loss = model(g.x.to(device), g.edge_index.to(device))

            print('Epoch: ', epoch, ', Iter: ', batch_idx, ', Loss: ', loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch}, Avg Loss: {avg_loss:.4f}')

        # Evaluate every 100 epochs
        # if epoch % 100 == 0:
        #     validity = evaluate(model)
        #     print(f'Evaluation - Validity: {validity:.2f}')
        #
        #     # Save best checkpoint
        #     if validity > best_validity:
        #         best_validity = validity
        #         checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch{epoch}.pth')
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'validity': validity
        #         }, checkpoint_path)
        #         print(f'Saved best checkpoint to {checkpoint_path}')

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')

    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--max_num_nodes', dest='max_num_nodes', type=int,
                        help='Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.')
    parser.add_argument('-e', '--epochs', type=int)

    parser.set_defaults(dataset='grid',
                        feature_type='id',
                        lr=0.001,
                        batch_size=1,
                        num_workers=1,
                        epochs=10,
                        max_num_nodes=-1)
    return parser.parse_args()


def main():
    args = arg_parse()

    from dataset import QuantumCircuitDataset, DataLoader

    graphs = QuantumCircuitDataset(
        base_num_samples=1,
        num_qubits=4,
        regenerate=False,
    )
    max_num_nodes = max([graphs[i]['g'].num_nodes for i in range(len(graphs))])

    graphs_train = graphs

    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(max_num_nodes))

    dataset_loader = DataLoader(graphs_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = GraphVAE(graphs_train[0]['g'].num_node_features,64, 256, max_num_nodes).to(device)

    train(args, dataset_loader, model)


if __name__ == '__main__':
    main()
