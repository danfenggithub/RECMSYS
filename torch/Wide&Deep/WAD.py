import pandas as pd
import torch
import sys
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 引入脚本文件
from datasets.src.inputs import SparseFeat, DenseFeat, get_feature_names
from datasets.src.models import WDL
from datasets.src.models.basemodel import BaseModel
from datasets.src.layers import DNN
from datasets.src.layers.activation import activation_layer
from datasets.src.inputs import combined_dnn_input

data = pd.read_csv('./datasets/data/criteo_sample.txt')
sparse_features = ['C' + str(i) for i in range(1, 27)]       #生成离散属性列名
dense_features = ['I' + str(i) for i in range(1, 14)]        #生成连续属性列名

data[sparse_features] = data[sparse_features].fillna('-1', ) #补全缺失值
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

import torch.nn as nn


class Linear(nn.Module):
    """Instantiates of Wide Module

    :param: feature_columns：包含模型线性部分所使用的所有特征的迭代器
    :param: feature_index: dict, 包含所需特征的索引，形如:OrderedDict: {feature_name:(start, start+dimension)}
    :param: init_std: floot, 初始化embedding向量时的标准差
    :param: device: str, 指明计算的设备
    """

    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device

        # 筛选出离散特征
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        # 筛选出连续特征
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        # 筛选出变长特征
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
        # 创建embedding字典
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        # 初始化embedding向量
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1)).to(
                device)
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X):

        # 得到离散、连续、变长特征的embedding向量列表
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        varlen_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        # 将离散特征和连续特征连接起来，然后对所有特征求和，得到Wide层的输出
        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            linear_sparse_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
            linear_dense_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit = linear_sparse_logit + linear_dense_logit
        elif len(sparse_embedding_list) > 0:
            linear_logit = torch.sum(
                torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        elif len(dense_value_list) > 0:
            linear_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
        else:
            linear_logit = torch.zeros([X.shape[0], 1])
        return linear_logit  # 形如[X.shape[0],1]


class WDL(BaseModel):
    """Instantiates the Wide&Deep architecture.

    :param linear_feature_columns: 包含模型线性部分所使用的所有特征的迭代器
    :param dnn_feature_columns: 包含模型深度部分所使用的所有特征的迭代器
    :param dnn_hidden_units: list,正整数列表或空列表, 每层深度网络的单元数
    :param l2_reg_linear: float.线性层的l2正则化项系数
    :param l2_reg_embedding: embedding层使用的正则化项系数
    :param l2_reg_dnn: 深度网络的正则化项系数
    :param init_std: float,用来初始化embedding层
    :param seed: integer ,随机种子
    :param dnn_dropout: float in [0,1), DNN中的dropout概率.
    :param dnn_activation: DNN激活函数
    :param dnn_use_bn: bool. 在DNN中是否加入BN层
    :param task: str, 指明为分类问题或是回归问题
    :param device: str, 指明计算的设备
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False,
                 task='binary', device='cpu'):

        super(WDL, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device)

        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0

        if self.use_dnn:
            # 初始化Deep层
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2_reg_dnn)

        self.to(device)

    def forward(self, X):

        # 得到离散特征embedding向量列表以及连续特征embedding向量列表
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        # 得到Wide层的输出
        logit = self.linear_model(X)

        if self.use_dnn:
            # 得到Deep层的输出
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        # 通过Sigmoid层，将logit转化为在[0,1]内的值，输出预测的分数
        y_pred = self.out(logit)

        return y_pred

# 3 模型的实例化
# 3.1 数据准备
# 3.1.1 输入数据编码
# 对离散特征进行标签编码，对连续特征进行简单转换
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 计算每个离散特征域的特征数，并记录连续特征域的名称
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
    linear_feature_columns + dnn_feature_columns)

# 3.1.2 训练测试集切分
# 为模型创建输入数据
train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

# 3.2 选择训练设备及实例化模型
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
               task='binary',
               l2_reg_embedding=1e-5, device=device)

# 4.模型训练
sys.stdout = open('./temp/train.log', mode = 'w',encoding='utf-8')

# 定义优化器及评测指标，选择adagrad作为优化器；同时，选择交叉熵和AUC作为评测指标评价模型的好坏。
model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

DEBUG = True
epoch = 2 if DEBUG else 50

# 根据参数训练模型
# 在criteo上训练时，部分deepCTR模型的表现可能会随着训练的轮次变差，并且单轮训练时间较长，所以这里epochs设置较小
model.fit(train_model_input, train[target].values, batch_size=32, epochs=epoch, verbose=2, validation_split=0.2)

# 5.模型预测
sys.stdout = open('./temp/test.log', mode = 'w',encoding='utf-8')
pred_ans = model.predict(test_model_input, 256)
print("")
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))