import torch
import numpy as np
import csv

h = np.load('dataset/h.npy')
h = torch.Tensor(h)

author_mapping = np.load('dataset/author_mapping.npy',
                         allow_pickle=True).item()

f_read = open('dataset/query_dataset.csv', 'r')
f_write = open('dataset/query_answer.csv', 'w', newline='')
wr = csv.writer(f_write)

true_cnt = 0
false_cnt = 0
for line in csv.reader(f_read):
    if line[0] == 'ID':
        wr.writerow(['ID', ' ID', ' label'])
    else:
        label = bool
        node1 = author_mapping[int(line[0])]
        node2 = author_mapping[int(line[1])]
        v1 = h[node1]
        v2 = h[node2]
        sim = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
        if sim > 0.5:
            label = True
            true_cnt += 1
        else:
            label = False
            false_cnt += 1
        wr.writerow([line[0], line[1], ' ' + str(label)])

f_read.close()
f_write.close()

print(true_cnt, false_cnt)
