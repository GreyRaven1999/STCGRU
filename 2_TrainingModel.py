import os
import random
import numpy as np
import torch
import torch.nn as nn
import NeuralNetworks as dl
import joblib
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    from torch.backends import cudnn
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True


if __name__ == "__main__":
    seed = 34
    total_fold = 10  # 10折
    '''深度学习超参数'''
    input_size = 16
    hidden_size = 128
    num_layers_lstm = 1
    num_layers_bilstm = 2
    num_classes = 2
    batch_size = 256
    num_epochs = 100
    # learning_rate = 0.0003
    learning_rate = 0.003

    # model_name = 'dl.STCGRU()'

    start = time.perf_counter()
    name = locals()
    seed_everything(seed)
    writer = SummaryWriter('./runs/Modified_STCGRU_with_seed_' + str(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # EEG = joblib.load('EEGdata/EEG.pkl')
    # label = joblib.load('EEGdata/label.pkl')
    # srate = joblib.load('EEGdata/srate.pkl')

    for i in range(total_fold):
        train_data_combine = torch.load("EEGData/TrainData/train_data_"
                                        + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        valid_data_combine = torch.load("EEGData/ValidData/valid_data_"
                                        + str(i + 1) + "_fold_with_seed_" + str(seed) + ".pth")
        '''定义深度学习模型'''
        model = dl.Modified_STCGRU().to(device)
        model_type = str(model.__class__)[23:-2]
        '''定义损失函数Loss 和 优化算法optimizer'''
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)  # 余弦退火
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
        print('开始第%d次训练，共%d次' % (i + 1, total_fold))

        # 生成迭代器，根据小批量数据大小划分每批送入模型的数据集
        train_loader = DataLoader(dataset=train_data_combine,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=0)
        valid_loader = DataLoader(dataset=valid_data_combine,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=0)
        total_step = len(train_loader)
        '''模型训练'''
        for epoch in range(num_epochs):
            '''训练'''
            model, optimizer = dl.model_training(writer, i, type='train', num_epochs=num_epochs,
                                                 epoch=epoch, loader=train_loader, neural_network=model,
                                                 criterion=criterion, optimizer=optimizer)
            '''验证'''
            optimizer, lr_list = dl.model_training(writer, i, type='validation', epoch=epoch,
                                                   loader=valid_loader, neural_network=model, criterion=criterion,
                                                   optimizer=optimizer, scheduler=scheduler)
        torch.save(model.state_dict(),
                   "ModelParameter/" + model_type + "/" +
                   model_type + "_" + str(i + 1) + "_fold_model_parameter_with_seed_" + str(seed) + ".pth")
        print(model_type + "模型第" + str(i + 1) + "次训练结果保存成功")
    end = time.perf_counter()
    print("训练及验证运行时间为", round(end - start), 'seconds')
