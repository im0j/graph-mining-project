import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs
from tqdm.auto import trange

# from gat import GAT
from dgl.nn.pytorch.conv import GATConv as GAT


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


g = load_graphs("dataset/paper_author_relationship.bin")[0][0].to(device)
author_mapping = np.load('dataset/author_mapping.npy', allow_pickle=True).item()

train = pd.read_csv(
    'dataset/train_dataset.csv',
    header=0,
    names=['source', 'target', 'label'],
    dtype={'label': bool},
    sep=',\s+',
    engine='python'
)
# train['source'] = train['source'].map(author_mapping)
# train['target'] = train['target'].map(author_mapping)

# create the model, 2 heads, each head has hidden size 8
net = GAT(in_feats=16,
          out_feats=16,
          num_heads=2).to(device)


# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# main loop
dur = []
for epoch in trange(1000):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    h = net(g, g.ndata['x'])
    h = torch.flatten(h, start_dim=1)

    similarities = torch.empty((1000, 1), requires_grad=False)
    for i in range(1000):
        data1 = author_mapping[train['source'][i]]
        data2 = author_mapping[train['target'][i]]
        v1 = h[data1]
        v2 = h[data2]
        sim = torch.norm(torch.dot(v1, v2))
        similarities[i] = sim
    target = torch.ones((1000, 1))
    loss = loss_fn(similarities, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 100 == 0:
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
