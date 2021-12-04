import numpy as np
import numpy.linalg as L
import pandas as pd

h = np.load('dataset/h_1024f.npy')

author_mapping = np.load('dataset/author_mapping.npy', allow_pickle=True).item()
query = pd.read_csv(
    'dataset/query_dataset.csv',
    header=0,
    names=['source', 'target'],
    sep=',\s+',
    engine='python'
)

threshold = 0.5
sims = []
labels = []

for author1, author2 in query.itertuples(index=False, name=None):
    # get node number of each author
    node1 = author_mapping[author1]
    node2 = author_mapping[author2]
    # get embedding of each node
    v1 = h[node1]
    v2 = h[node2]
    # compute similarity
    sim = np.dot(v1, v2) / (L.norm(v1) * L.norm(v2))
    sims.append(sim)
    # perform classification
    label = sim >= 0.5
    labels.append(label)

query['sims'] = sims
query['labels'] = labels

# print statistics
pos_idx = query['labels'] == True
num_pos = sum(pos_idx)
pos_sims = query[pos_idx]['sims']
max_pos_sims = pos_sims.max()
min_pos_sims = pos_sims.min()
avg_pos_sims = pos_sims.mean()

neg_idx = query['labels'] == False
num_neg = sum(neg_idx)
neg_sims = query[neg_idx]['sims']
max_neg_sims = neg_sims.max()
min_neg_sims = neg_sims.min()
avg_neg_sims = neg_sims.mean()

print(f"""
# Stats
- Positive labels: {num_pos}
  - Max sim      : {max_pos_sims: .4f}
  - Avg sim      : {avg_pos_sims: .4f}
  - Min sim      : {min_pos_sims: .4f}
- Negative labels: {num_neg}
  - Max sim      : {max_neg_sims: .4f}
  - Avg sim      : {avg_neg_sims: .4f}
  - Min sim      : {min_neg_sims: .4f}
""".strip())

query.to_csv(
    'dataset/query_answer.csv',
    columns=['source', 'target', 'labels'],
    index=False
)
