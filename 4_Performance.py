import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 整张图figure的标题自定义设置
'''
plt.suptitle('整张图figure的标题：suptitle',  # 标题名称
             x=0.5,  # x轴方向位置
             y=0.98,  # y轴方向位置
             size=15,  # 大小
             ha='center',  # 水平位置，相对于x,y，可选参数：{'center', 'left', right'}, default: 'center'
             va='top',  # 垂直位置，相对于x,y，可选参数：{'top', 'center', 'bottom', 'baseline'}, default: 'top'
             weight='bold',  # 字体粗细，以下参数可选
             # 其它可继承自matplotlib.text的属性
             # 标题也是一种text，故可使用text的属性，所以这里只是展现了冰山一角
             rotation=1,  ##标题旋转，传入旋转度数，也可以传入vertical', 'horizontal'
             )
'''


def ROC_curve(y_label, y_pre, number):
    font1 = {'family': 'Times New Roman',
             'style': 'italic',
             'weight': 'normal',
             'size': 13,
             }
    fpr, tpr, thresholds = roc_curve(y_label, y_pre, pos_label=0)

    # for i, value in enumerate(thresholds):
    #     print("%f %f %f" % (fpr[i], tpr[i], value))
    roc_auc = auc(fpr, tpr)
    if number == 1:
        plt.figure(figsize=(13, 6), facecolor='w', dpi=100)
        plt.suptitle('ROC Curve of Ablation Experiment',
                     fontproperties='Times New Roman',
                     size=20,  # 大小
                     y=0.95,  # y轴方向位置
                     )
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='dodgerblue', label='Variant 1 (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.legend(loc="lower right",
                   prop=font1,
                   )  # 添加图例到右下角
        plt.xticks(fontproperties='Times New Roman',
                   size=11)
        plt.yticks(fontproperties='Times New Roman',
                   size=11)
        plt.xlim([0, 1])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([0, 1])
        # plt.plot([0, 1], [0, 1], 'r--')  # 对角线
        plt.xlabel('False Positive Rate',
                   fontproperties='Times New Roman',
                   size=16,  # 大小
                   )
        plt.ylabel('True Positive Rate',
                   fontproperties='Times New Roman',
                   size=16,  # 大小
                   )
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='dodgerblue', label='Variant 1 (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.xticks(fontproperties='Times New Roman',
                   size=11)
        plt.yticks(fontproperties='Times New Roman',
                   size=11)
        plt.xlim([0, 0.10])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([0.90, 1])
        plt.xlabel('False Positive Rate',
                   fontproperties='Times New Roman',
                   size=16,  # 大小
                   )
        plt.ylabel('True Positive Rate',
                   fontproperties='Times New Roman',
                   size=16,  # 大小
                   )
        # plt.show()
    if number == 2:
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='springgreen', label='Variant 2 (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.legend(loc="lower right",
                   prop=font1,
                   )  # 添加图例到右下角
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='springgreen', label='Variant 2 (area = {0:.4f})'.format(roc_auc), lw=2)
        # plt.show()
    if number == 3:
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='tomato', label='Variant 3 (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.legend(loc="lower right",
                   prop=font1,
                   )  # 添加图例到右下角
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='tomato', label='Variant 3 (area = {0:.4f})'.format(roc_auc), lw=2)
        # plt.show()
    if number == 4:
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', label='Variant 4 (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.legend(loc="lower right",
                   prop=font1,
                   )  # 添加图例到右下角
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', label='Variant 4 (area = {0:.4f})'.format(roc_auc), lw=2)
        # plt.show()
    if number == 5:
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='gold', label='STCGRU (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.legend(loc="lower right",
                   prop=font1,
                   )  # 添加图例到右下角
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='gold', label='STCGRU (area = {0:.4f})'.format(roc_auc), lw=2)
        plt.show()


seed = 34

model_type = 'BiGRU'
labels = joblib.load('Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
results = joblib.load('Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
ROC_curve(labels, results, number=1)

model_type = 'BiGRU_and_SCNN'
labels = joblib.load('Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
results = joblib.load('Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
ROC_curve(labels, results, number=2)

model_type = 'STCGRU_without_STCNN'
labels = joblib.load('Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
results = joblib.load('Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
ROC_curve(labels, results, number=3)

model_type = 'Modified_STCGRU'
labels = joblib.load('Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
results = joblib.load('Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
ROC_curve(labels, results, number=4)

model_type = 'STCGRU'
labels = joblib.load('Result/' + model_type + '/labels_with_seed_' + str(seed) + '.pkl')
results = joblib.load('Result/' + model_type + '/results_with_seed_' + str(seed) + '.pkl')
ROC_curve(labels, results, number=5)