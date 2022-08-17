import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

DEBUG = True
continous_features = 13  # 数据集中的连续特征数量
# 训练集、验证集中数据的下标最大值，可视自身运行环境而定，数据集中共计80000条数据
Num_train = 10000
Num_valid = 13000
use_cuda = False  # 是否使用cuda
epochs = 20


class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, :-1].values

    def __getitem__(self, idx):
        dataI, targetI = self.train_data[idx, :], self.target[idx]
        # index of continous features are zero
        Xi_coutinous = np.zeros_like(dataI[:continous_features])
        Xi_categorial = dataI[continous_features:]
        Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)

        # value of categorial features are one (one hot features)
        Xv_categorial = np.ones_like(dataI[continous_features:])
        Xv_coutinous = dataI[:continous_features]
        Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
        return Xi, Xv, targetI

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)


class DeepFM(nn.Module):

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        """
        初始化工作

        输入:
        - feature_size: 所有特征可能的取值种类.
        - embedding_size: embedding维度.
        - hidden_dims: DNN的维度.
        - num_classes: 预测问题的类别.
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.randn(1))
        # 检查是否使用cuda
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # fm初始化
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

        # dnn初始化
        all_dims = [self.field_size * self.embedding_size] + \
                   self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))

    def forward(self, Xi, Xv):
        """
        输入:
        - Xi: 输入的索引, shape ：(N, field_size, 1)
        - Xv: 输入的值, shape ：(N, field_size, 1)
        """
        # fm部分
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                  enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                   enumerate(self.fm_second_order_embeddings)]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        # dnn部分
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        # 对fm和dnn的预测求和
        total_sum = torch.mean(fm_first_order, 1) + \
                    torch.mean(fm_second_order, 1) + torch.mean(deep_out, 1) + self.bias

        return total_sum


feature_sizes = np.loadtxt('./datasets/data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]

model = DeepFM(feature_sizes, use_cuda=False)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

train_data = CriteoDataset('./datasets/data', train=True)
loader_train = DataLoader(train_data, batch_size=200, sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = CriteoDataset('./datasets/data', train=True)
loader_val = DataLoader(val_data, batch_size=3000, sampler=sampler.SubsetRandomSampler(range(Num_train, 13000)))


def check_accuracy(loader, model):
    global f
    if loader.dataset.train:
        f.write('Checking accuracy on validation set\n')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for xi, xv, y in loader:
            xi = xi.to(device=model.device, dtype=torch.long)  # move to device, e.g. GPU
            xv = xv.to(device=model.device, dtype=torch.float)
            y = y.to(device=model.device, dtype=torch.bool)
            # print(y.shape)
            total = model(xi, xv)
            preds = (torch.sigmoid(total) > 0.5)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        f.write('Got %d / %d correct (%.2f%%)\n' % (num_correct, num_samples, 100 * acc))

        return acc


def fit(model, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
    global f
    model = model.train().to(device=model.device)
    criterion = F.binary_cross_entropy_with_logits
    Loss_epoch = []
    accuracy_epoch = []
    for epoch in range(epochs):
        iter = 0
        Loss = 0
        for t, (xi, xv, y) in enumerate(loader_train):
            iter += 1
            xi = xi.to(device=model.device, dtype=torch.long)
            xv = xv.to(device=model.device, dtype=torch.float)
            y = y.to(device=model.device, dtype=torch.float)
            total = model(xi, xv)
            loss = criterion(total, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss += (loss.item() / y.shape[0])

        Loss_epoch.append(Loss / iter)
        if verbose:
            f.write('\nEpoch %d, loss = %.4f\n' % (epoch, Loss_epoch[-1]))
            accuracy_epoch.append(check_accuracy(loader_val, model))

    return Loss_epoch, accuracy_epoch


feature_sizes = np.loadtxt('./datasets/data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

Time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
print('training...')
with open('./work/{}.log'.format(Time), 'w') as f:
    Loss_epoch, accuracy_epoch = fit(model, loader_train, loader_val, optimizer, epochs=epochs, verbose=True)

print('Done.')

plt.figure()  # 初始化画布
x1 = range(0, epochs)  # 取横坐标的值
plt.plot(x1, Loss_epoch, label='loss')  # 绘制折线图
plt.scatter(x1, Loss_epoch)  # 绘制散点图
plt.xlabel('Epoch #')  # 设置坐标轴名称
plt.ylabel('LOSS')
plt.legend()
plt.show()  # 显示图片

plt.figure()  # 初始化画布
x1 = range(0, epochs)  # 取横坐标的值
plt.plot(x1, accuracy_epoch, label='accuracy')
plt.scatter(x1, accuracy_epoch)
plt.xlabel('Epoch #')  # 设置坐标轴名称
plt.ylabel('Accuracy')
plt.legend()
plt.show()  # 显示图片
