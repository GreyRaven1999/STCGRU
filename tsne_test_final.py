import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import NeuralNetworks as dl
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
from time import time


class T_sne_visual():
    def __init__(self, model,  dataloader):
        self.model = model
        self.dataloader = dataloader

    def visual_feature_map(self, layer):
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).register_forward_hook(self.forward_hook)
            for data, label in self.dataloader:
                data = data.to(device)
                label = label.to(device)
                self.model(data.float())
                labels.append(label.cpu().numpy())
            self.feature_map_list = torch.cat(self.feature_map_list, dim=0)
            # self.feature_map_list = torch.flatten(self.feature_map_list, start_dim=1)
            labels = [i for list in labels for i in list]
            self.t_sne(np.array(self.feature_map_list.cpu().numpy()), labels, title=f'{layer} feature map\n')

    def forward_hook(self, model, input, output):
        
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time, title):
        plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        plt.legend(title='')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label, title):
        # t-sne处理
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # 绘图
        # plt.scatter(data[:, 0], data[:, 1])
        # plt.title('t-SNE Visualization of PyTorch Model Features')
        sns.scatterplot(x='x', y='y', hue='label', s=3, data=df)
        self.set_plt(start_time, end_time, title)
        # plt.savefig('1.jpg', dpi=400)
        plt.show()


seed = 220
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'STCGRU'
model = dl.STCGRU()
i = 0
model_loc = "ModelParameter/" + model_type + "/" + model_type + "_" + \
            str(i + 1) + "_fold_model_parameter_with_seed_" + str(seed) + ".pth"
model_dict = torch.load(model_loc)
model.load_state_dict(model_dict)

for child in model.children():
    print(child)

# 剔除最后的全连接层

new_model = nn.Sequential(*list(model.children())[:-1])

new_model.eval()  # 设置为评估模式
new_model = new_model.to(device)

test_data_combine = torch.load("EEGdata/testdata/test_data_"
                               + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
test_loader = DataLoader(dataset=test_data_combine,
                         batch_size=256,
                         shuffle=True,
                         drop_last=False,
                         pin_memory=True,
                         num_workers=0)


t = T_sne_visual(model, test_loader)
t.visual_feature_map('flatten')


features = []
for step, (data, labels) in enumerate(test_loader):
    data = data.to(device)
    labels = labels.to(device)
    # 前向传播
    with torch.no_grad():
        feature = new_model(data.float())
    features.append(feature.squeeze().cpu().numpy())

# 将提取的特征堆叠为一个数组
features = np.vstack(features)
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(features)

# 可视化t-SNE结果
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('t-SNE Visualization of PyTorch Model Features')
plt.show()
