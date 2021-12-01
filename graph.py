import dgl
import torch as th
from dgl.data.utils import save_graphs

g1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
save_graphs("/home/leedongwook/PycharmProjects/ai607_project/g1", g1)





