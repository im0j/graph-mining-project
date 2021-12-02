import os.path
from itertools import combinations

import dgl
import numpy as np
import torch
from torch import nn

all_authors = set()

if os.path.isfile('dataset/paper_author_relationship.bin'):
    graph = dgl.load_graphs('dataset/paper_author_relationship.bin')[0][0]
else:
    with open('dataset/paper_author_relationship.csv') as f:
        for paper, authors in enumerate(f):
            all_authors.update(map(int, authors.split(',')))

        author_mapping = {k: v for k, v in zip(all_authors, range(len(all_authors)))}
        np.save('dataset/author_mapping.npy', author_mapping)

        num_authors = len(all_authors)

        f.seek(0)

        edges = set()
        for paper, authors in enumerate(f):
            authors = map(int, authors.split(','))
            authors = sorted(author_mapping[author] for author in authors)
            clique = combinations(authors, r=2)
            edges.update(clique)

        graph = dgl.graph(tuple(zip(*edges)))
        graph = dgl.add_self_loop(graph)
        graph = dgl.to_bidirected(graph)

node_features = 16
graph.ndata['x'] = nn.Parameter(
    torch.Tensor(graph.num_nodes(), node_features))
nn.init.uniform_(graph.ndata['x'], -1, 1)
dgl.save_graphs('dataset/paper_author_relationship.bin', [graph])
