from dgl.data.utils import load_graphs
import time
import numpy as np
import torch
import torch.nn.functional as F
from gat import GAT
import pandas as pd

g = load_graphs("dataset/paper_author_relationship.bin")[0][0]

train = pd.read_csv(
    'dataset/train_dataset.csv',
    header=0,
    names=['source', 'target', 'label'],
    dtype={'label': bool},
    sep=',\s+',
    engine='python'
)


# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=16,
          hidden_dim=8,
          out_dim=16,
          num_heads=2)


# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

author_mapping = np.load('dataset/author_mapping.npy', allow_pickle='TRUE').item()


# main loop
dur = []
for epoch in range(1000):
    if epoch >= 3:
        t0 = time.time()

    '''
    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])
    '''

    loss = 0
    margin = 0
    for i in range(1000):
        data1 = author_mapping[train['source'][i]]
        data2 = author_mapping[train['target'][i]]
        z1 = g.ndata['x'][data1]
        z2 = g.ndata['x'][data2]
        sim = torch.dot(z1, z2)
        loss += max(0, -sim + margin)

    loss = torch.tensor(loss, requires_grad=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    if epoch % 100 == 0:
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))

