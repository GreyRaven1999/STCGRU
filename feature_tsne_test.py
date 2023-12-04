import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np


def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import NeuralNetworks as dl
import joblib
from sklearn.manifold import TSNE


def test(i, loader, neural_network):
    acc = 0
    results_sum = []
    labels_test_sum = []
    results_PR_sum = []
    for step, (data, labels) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = neural_network(data.float())
        acc += outputs.argmax(dim=1).eq(labels).type_as(torch.FloatTensor()).mean()
        results_sum = np.append(results_sum, outputs.argmax(dim=1).cpu().numpy())
        labels_test_sum = np.append(labels_test_sum, labels.cpu().numpy())
        results_PR_sum.extend(outputs.detach().cpu().numpy())
    acc_average = acc / (step + 1)
    print("第" + str(i + 1) + "次训练测试集准确率: {:.4f}".format(acc_average))
    return acc_average, results_sum, labels_test_sum, results_PR_sum


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    # 定义一种字体属性
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18}
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(num=None, figsize=(6, 6), dpi=60)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict=font1, y=1.05)
    plt.colorbar(shrink=0.64)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontdict=font2)
    plt.yticks(tick_marks, classes, rotation=45, fontdict=font2)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 verticalalignment="center",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontdict=font2)
    plt.tight_layout()
    plt.ylabel('Actual Class', fontdict=font2)
    plt.xlabel('Predict Class', fontdict=font2)
    plt.subplots_adjust(left=0.2, top=1.02, bottom=0.05)
    plt.show()


# 定义钩子函数
def hook_fn(module, input, output):
    # 在这里可以获取特定层的输出
    extracted_features.append(output)


if __name__ == "__main__":
    name = locals()
    seed = 220
    dl.seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'STCGRU'

    batch_size = 256
    total_fold = 10  # 10折

    test_acc_sum = 0
    results_sum = []
    labels_test_sum = []
    results_PR_sum = []

    for i in range(total_fold):
        test_data_combine = torch.load("EEGdata/testdata/test_data_"
                                       + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        test_loader = DataLoader(dataset=test_data_combine,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=False,
                                 pin_memory=True,
                                 num_workers=0)

        model = dl.STCGRU()
        model_loc = "ModelParameter/" + model_type + "/" + model_type + "_" + \
                    str(i + 1) + "_fold_model_parameter_with_seed_" + str(seed) + ".pth"
        model_dict = torch.load(model_loc)
        model.load_state_dict(model_dict)
        model = model.to(device)
        # 选择要提取特征的层
        target_layer = model.flatten

        # 注册钩子
        extracted_features = []
        hook = target_layer.register_forward_hook(hook_fn)

        '''测试'''
        acc_average, results, labels_test, results_PR = test(i, loader=test_loader, neural_network=model)
        combined_tensor = torch.cat(extracted_features)
        combined_tensor = combined_tensor.to('cpu')
        combined_tensor = combined_tensor.detach().numpy()

        label = test_data_combine.tensors[1]
        label = label.to('cpu')
        label = label.numpy()
        fig = plt.figure(figsize=(10, 10))

        plotlabels(visual(combined_tensor), label, '(a)')

        plt.legend()
        plt.show(fig)
