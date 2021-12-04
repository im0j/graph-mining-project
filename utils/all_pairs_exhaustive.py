from heapq import nlargest

import numpy as np
import numpy.linalg as L
from tqdm.auto import tqdm

h = np.load('dataset/h.npy')
author_mapping = np.load('dataset/author_mapping.npy', allow_pickle=True).item()
node_mapping = {v: k for k, v in author_mapping.items()}


def stream_sims():
    num_nodes = len(node_mapping)
    num_combinations = num_nodes * (num_nodes-1) // 2
    norms = L.norm(h, axis=1)
    with tqdm(total=num_combinations) as t:
        for node1 in range(num_nodes):
            t.postfix = f'{node1}/{num_nodes}'
            v1 = h[node1]
            norm1 = norms[node1]
            for node2 in range(node1+1, num_nodes):
                v2 = h[node2]
                norm2 = norms[node2]
                sim = np.dot(v1, v2) / (norm1 * norm2)
                yield (sim, node1, node2)
                t.update()


def main():
    top_1k = nlargest(10000, stream_sims())
    with open('top_1k_pairs.csv', 'wt') as f:
        f.write('id,id\n')
        for sim, node1, node2 in top_1k:
            f.write(f"{node_mapping[node1]},{node_mapping[node2]}\n")


if __name__ == '__main__':
    main()
