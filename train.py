import argparse
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.utils import load_graphs
from dgl.nn.pytorch.conv import GATConv as GAT


parser = argparse.ArgumentParser()
parser.add_argument('--thres', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


g = load_graphs("dataset/paper_author_relationship.bin")[0][0].to(device)
author_mapping = np.load('dataset/author_mapping.npy',
                         allow_pickle=True).item()

train = pd.read_csv(
    'dataset/train_dataset.csv',
    header=0,
    names=['source', 'target', 'label'],
    dtype={'label': bool},
    sep=',\s+',
    engine='python'
)
val = pd.read_csv(
    'dataset/valid_dataset.csv',
    header=0,
    names=['source', 'target', 'label'],
    dtype={'label': bool},
    sep=',\s+',
    engine='python'
)

# create graph node embeddings
node_features = 32  # TODO: hyperparameter
num_nodes = g.num_nodes()
g.ndata['x'] = nn.Parameter(torch.Tensor(
    num_nodes, node_features)).to(device)
nn.init.uniform_(g.ndata['x'], -1, 1)

# create the model, 2 heads, each head has hidden size 8
net = GAT(
    in_feats=node_features,
    out_feats=node_features,
    num_heads=16  # TODO: hyperparameter
).to(device)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_fn = nn.MSELoss().to(device)

# main loop
dur = []
for epoch in range(args.epochs):
    t0 = time.time()

    h = net(g, g.ndata['x'])
    h = torch.flatten(h, start_dim=1)

    # Validation
    net.train(False)

    val_out = torch.empty((1000, 1), requires_grad=False)
    val_label = torch.tensor(val['label']).to(torch.float).reshape((1000, 1))
    for i in range(1000):
        data1 = author_mapping[val.at[i, 'source']]
        data2 = author_mapping[val.at[i, 'target']]
        v1 = h[data1]
        v2 = h[data2]
        sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        val_out[i] = sim

    val_loss = loss_fn(val_out, val_label)
    val_pred = val_out > args.thres
    val_acc = torch.mean((val_pred == val_label).to(torch.float)).detach().float()

    # Training
    net.train(True)

    pos_count = 64  # TODO: hyperparameter
    pos_out = torch.empty((pos_count, 1), requires_grad=False)
    pos_label = torch.ones((pos_count, 1))
    random_samples = np.random.randint(0, 1000-1, pos_count)
    for i in range(pos_count):
        data1 = author_mapping[train.at[random_samples[i], 'source']]
        data2 = author_mapping[train.at[random_samples[i], 'target']]
        v1 = h[data1]
        v2 = h[data2]
        sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        pos_out[i] = sim

    neg_count = 192  # TODO: hyperparameter
    neg_out = torch.empty((neg_count, 1), requires_grad=False)
    neg_label = torch.zeros((neg_count, 1))
    random_samples = np.random.randint(0, num_nodes-1, size=(neg_count, 2))
    for i in range(neg_count):
        data1 = random_samples[i, 0]
        data2 = random_samples[i, 1]
        v1 = h[data1]
        v2 = h[data2]
        sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        neg_out[i] = sim

    train_out = torch.cat([pos_out, neg_out])
    train_label = torch.cat([pos_label, neg_label])
    train_loss = loss_fn(train_out, train_label)
    train_pred = train_out > args.thres
    train_acc = torch.mean((train_pred == train_label).to(torch.float)).detach().float()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Logging
    dur.append(time.time() - t0)

    print(
        f'Epoch {epoch: 5d}',
        f'Train loss {train_loss:.4f}',
        f'Val loss {val_loss:.4f}',
        f'Train acc {train_acc:.3%}',
        f'Val acc {val_acc:.3%}',
        f'Time/it {np.mean(dur):.4f}',
        sep=' | ',
        end=('\n' if epoch % 100 == 0 else '\r')
    )
