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
    neural_network.eval()
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
        # 使用 t-SNE 进行降维和可视化
        tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
        tsne_result = tsne.fit_transform(combined_tensor)
        print(tsne_result.shape)  # [num, 2]

        tsne_result_min, tsne_result_max = tsne_result.min(0), tsne_result.max(0)

        tsne_result_final = (tsne_result - tsne_result_min) / (tsne_result_max - tsne_result_min)

        label = test_data_combine.tensors[1]
        label = label.to('cpu')
        label = label.numpy()
        # 绘制可视化结果
        for j in np.unique(label):
            plt.scatter(tsne_result_final[label == j, 0], tsne_result_final[label == j, 1], label=str(j), s=2)
        plt.title('PCA + t-SNE Visualization')
        plt.legend()
        # if i == total_fold-1:
        #     plt.legend()
        #     plt.show()
        plt.show()
        results_sum = np.append(results_sum, results)
        labels_test_sum = np.append(labels_test_sum, labels_test)
        results_PR_sum.extend(results_PR)
        name['test_acc_average_' + str(i + 1)] = acc_average
        confusion_matrix_single = confusion_matrix(labels_test, results, labels=[0, 1])
        kappa_single = cohen_kappa_score(labels_test, results)
        sensitivity_single = confusion_matrix_single[0, 0] / (confusion_matrix_single[0, 0] + confusion_matrix_single[0, 1])  # 灵敏度（召回率）
        specificity_single = confusion_matrix_single[1, 1] / (confusion_matrix_single[1, 1] + confusion_matrix_single[1, 0])  # 特异度
        precision_single = confusion_matrix_single[0, 0] / (confusion_matrix_single[0, 0] + confusion_matrix_single[1, 0])  # 查准率
        F1_single = 2 * precision_single * sensitivity_single / (precision_single + sensitivity_single)  # F1值
        print(model_type + " 第%d折交叉验证测试集准确率: %.4f，kappa值：%.4f，灵敏度：%.4f，特异度：%.4f，查准率：%.4f，F1值：%.4f"
              % (i+1, acc_average, kappa_single, sensitivity_single, specificity_single, precision_single, F1_single))

    for i in range(total_fold):
        if i == 0:
            test_acc_sum = name['test_acc_average_' + str(i + 1)]
        else:
            test_acc_sum = np.append(test_acc_sum, name['test_acc_average_' + str(i + 1)])
        del name['test_acc_average_' + str(i + 1)]

test_acc_final = np.sum(test_acc_sum) / total_fold
test_acc_std = float(np.std(test_acc_sum))
confusion_matrix = confusion_matrix(labels_test_sum, results_sum, labels=[0, 1])

kappa = cohen_kappa_score(labels_test_sum, results_sum)
sensitivity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])  # 灵敏度（召回率）
specificity = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])  # 特异度
precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])  # 查准率
F1 = 2 * precision * sensitivity / (precision + sensitivity)  # F1值
# plot_confusion_matrix(confusion_matrix, classes=['MCI', 'HC'], normalize=False,
#                       title='Confusion matrix of ' + model_type)
plot_confusion_matrix(confusion_matrix, classes=['MCI', 'HC'], normalize=False,
                      title='Confusion Matrix of Variant 4')

print(model_type + " %d折交叉验证平均测试集准确率: %.4f ± %.4f，kappa值：%.4f，灵敏度：%.4f，特异度：%.4f，查准率：%.4f，F1值：%.4f"
      % (total_fold, test_acc_final, test_acc_std, kappa, sensitivity, specificity, precision, F1))

results_PR_sum_single = np.zeros(np.size(results_PR_sum, 0))
for i in range(len(results_PR_sum)):
    results_PR_sum_single[i] = results_PR_sum[i][0]
joblib.dump(labels_test_sum, 'Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
joblib.dump(results_PR_sum_single, 'Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
