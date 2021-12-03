import numpy as np
import torch
from tqdm import trange
import csv

h = np.load('dataset/h.npy')
h = torch.Tensor(h)

author_mapping = np.load('dataset/author_mapping.npy',
                         allow_pickle=True).item()

node_mapping = dict([(value, key) for key, value in author_mapping.items()])


already_true = set()
f_read = open('dataset/query_dataset.csv', 'r')
for line in csv.reader(f_read):
    if line[0] == 'ID':
        continue
    else:
        pair_list = sorted([int(line[0]), int(line[1])])
        pair = tuple(pair_list)
        already_true.add(pair)
f_read = open('dataset/train_dataset.csv', 'r')
for line in csv.reader(f_read):
    if line[0] == 'ID':
        continue
    else:
        pair_list = sorted([int(line[0]), int(line[1])])
        pair = tuple(pair_list)
        already_true.add(pair)
f_read = open('dataset/valid_dataset.csv', 'r')
for line in csv.reader(f_read):
    if line[0] == 'ID':
        continue
    else:
        pair_list = sorted([int(line[0]), int(line[1])])
        pair = tuple(pair_list)
        already_true.add(pair)


same_author = set()
cnt = 0

while cnt != 1000:
    random_samples1 = np.random.randint(0, 61441, size=1)
    random_samples2 = np.random.randint(0, 61441, size=1)
    if random_samples1 != random_samples2:
        v1 = h[random_samples1[0]]
        v2 = h[random_samples2[0]]
        sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        same_list = sorted([node_mapping[random_samples1[0]], node_mapping[random_samples2[0]]])
        if sim > 0.8 and tuple(same_list) not in already_true:
            same_author.add(tuple(same_list))
            cnt = len(same_author)
            print(cnt)
