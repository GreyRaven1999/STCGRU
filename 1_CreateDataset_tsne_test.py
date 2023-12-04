import os
import random
import scipy.io as sio
import numpy as np
import math
import mne
import joblib
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import NeuralNetworks as dl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def import_data(filePath_function, duration_function, label_single_function, type_function='preprocessed'):
    if type_function == 'raw':
        raw = mne.io.read_raw_edf(filePath_function[0], verbose='WARNING')
        raw.crop(tmin=0, tmax=1800)
        channel = raw.info.get('nchan')
        srate_function = raw.info.get('sfreq')
    else:
        raw = sio.loadmat(filePath_function[0], uint16_codec='latin1')
        data = raw.get('data')
        channel = np.size(data, 0)
        srate_function = raw.get('srate_function')
    EEG_function = np.zeros((channel, duration_function, 1))
    label_function = []
    for i in range(0, len(filePath_function)):
        if type_function == 'raw':
            raw = mne.io.read_raw_edf(filePath_function[i], verbose='WARNING')
            raw.crop(tmin=0, tmax=1800)
            data = raw.load_data()._data
            channel = raw.info.get('nchan')
        else:
            raw = sio.loadmat(filePath_function[i], uint16_codec='latin1')
            data = raw.get('data')
            channel = np.size(data, 0)
        if np.size(data, 1) > duration_function:
            epochs = math.floor(np.size(data, 1) / duration_function)
            data_new = np.zeros((channel, duration_function, epochs))
            for j in range(0, epochs):
                data_new[:, :, j] = data[:, duration_function * j: duration_function * (j + 1)]
        else:
            data_new = data
            epochs = np.size(data_new, 3)
        EEG_function = np.concatenate((EEG_function, data_new), axis=2)
        label_new = np.zeros(epochs) + label_single_function[i]
        label_function = np.append(label_function, label_new)
    EEG_function = np.delete(EEG_function, 0, axis=2)
    print('共有 %d 个片段，其中正常人有 %d 个，轻度认知障碍有 %d 个\n' % (
    np.size(EEG_function, 2), np.sum(label_function == 1), np.sum(label_function == 0)))
    return EEG_function, label_function, srate_function


if __name__ == "__main__":
    seed = 34
    dl.seed_everything(seed)
    '''EEG数据参数'''
    duration = 1280
    '''输入数据位置和标签'''
    filePath = ['D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/1.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/2.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/3.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/4.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/5.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/6.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/7.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/8.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/9.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/10.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/11.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/12.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/13.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/14.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/15.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/16.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/17.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/18.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/19.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/20.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/21.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/22.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/23.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/24.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/25.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/26.edf',
                'D:/BaiduSyncdisk/燕山大学/脑电数据/伊朗伊斯法罕Sina医院和Nour医院/27.edf']
    label_single = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
                            dtype=int)
    '''导入数据'''
    EEG, label, srate = import_data(filePath, duration, label_single, type_function='raw')
    # joblib.dump(EEG, 'EEGdata/EEG.pkl')
    # joblib.dump(label, 'EEGdata/label.pkl')
    # joblib.dump(srate, 'EEGdata/srate.pkl')

    EEG = np.transpose(EEG[0,:,:])  # 0 6 8 9 11 12 13 14 16 17 18有点奇怪

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(EEG)
    print(data_tsne.shape)  # [num, 2]

    data_tsne_min, data_tsne_max = data_tsne.min(0), data_tsne.max(0)

    data_tsne_final = (data_tsne - data_tsne_min) / (data_tsne_max - data_tsne_min)
    # 绘制降维后的数据点
    plt.figure(figsize=(8, 8))
    for j in np.unique(label):
        plt.scatter(data_tsne_final[label == j, 0], data_tsne_final[label == j, 1], label=str(j))
    plt.legend()
    plt.show()

    '''划分训练集和测试集'''
    total_fold = 10  # 10折
    [train_index, test_index] = dl.Split_Sets(total_fold, EEG.transpose(2, 1, 0))

    '''训练集和验证集处理'''
    for i in range(total_fold):
        train_data = EEG[:, :, train_index[i]]  # 得到初始训练数据
        train_data_label = label[train_index[i]]
        '''对训练集进行进一步划分，分为训练子集和验证子集'''
        train_data, valid_data, train_data_label, valid_data_label = train_test_split(train_data.transpose(2, 1, 0),
                                                                                      train_data_label, test_size=0.1)
        # 交换维度+变成张量
        train_data = torch.from_numpy(train_data)  # 变为片段个数,时长,通道个数
        train_data_label = (torch.from_numpy(train_data_label)).long()
        valid_data = torch.from_numpy(valid_data)  # 变为片段个数,时长,通道个数
        valid_data_label = (torch.from_numpy(valid_data_label)).long()
        '''测试集处理'''
        test_data = EEG[:, :, test_index[i]]
        test_data_label = label[test_index[i]]
        # 交换维度+变成张量
        test_data = torch.from_numpy(test_data.transpose(2, 1, 0))  # 变为片段个数,时长,通道个数
        test_data_label = (torch.from_numpy(test_data_label)).long()
        '''数据集在输入模型前的处理'''
        # 组合数据和标签
        train_data_combine = TensorDataset(train_data, train_data_label)
        valid_data_combine = TensorDataset(valid_data, valid_data_label)
        test_data_combine = TensorDataset(test_data, test_data_label)

        torch.save(train_data_combine,
                   "EEGData/TrainData/train_data_" + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        torch.save(valid_data_combine,
                   "EEGData/ValidData/valid_data_" + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        torch.save(test_data_combine,
                   "EEGData/TestData/test_data_" + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        print("第" + str(i + 1) + "折数据保存成功")
