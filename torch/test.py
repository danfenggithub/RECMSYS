# from scipy import sparse as csr
# import numpy as np
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# coo = csr.csr_matrix((data, (row, col)), shape=(4, 4))
# print(coo.todense())
import torch
from torch import nn

embedding = nn.Embedding(5, 2) # 假定字典中只有5个词，词向量维度为4
word = [[1, 2, 3],
        [2, 3, 4]] # 每个数字代表一个词，例如 {'!':0,'how':1, 'are':2, 'you':3,  'ok':4}
embed = embedding(torch.LongTensor(word))
print(embed)
print(embed.size())