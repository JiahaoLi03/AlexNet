import torch
import time
import copy
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import AlexNet


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=2)

    return train_dataloader, val_dataloader


# 训练过程
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备 有 GPU 使用 GPU 没有使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用 Adam 优化器, 学习率为 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()

    # 将模型放入到训练设备中
    model = model.to(device)

    # 复制当前模型的参数
    best_model = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精准度
    best_acc = 0.0

    # 训练集损失列表
    train_loss_all = []

    # 训练集准确度列表
    train_acc_all = []

    # 验证集损失列表
    val_loss_all = []

    # 验证集准确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):

        # 打印轮次信息
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-----------------------------------------------")

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0

        # 训练集准确度
        train_correct = 0.0

        # 验证集损失函数
        val_loss = 0.0

        # 验证集准确度
        val_correct = 0.0

        # 训练集样本数量
        train_num = 0

        # 验证集样本数量
        val_num = 0

        # 对每一个 mini-batch 训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)

            # 将标签放入到训练设备中
            b_y = b_y.to(device)

            # 设置模型为训练模式
            model.train()

            # 前向传播过程 输入为一个 batch， 输出为一个 batch 中对应的预测
            output = model(b_x)

            # 预测类别的下标
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个 batch_size 输出为一个 batch 中对应的预测
            loss = criterion(output, b_y)

            # 将梯度初始化为 0
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低 loss 函数计算值的作用
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)

            # 如果预测正确，则准确度 train_corrects + 1
            train_correct += torch.sum(pre_lab == b_y.data)

            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为评估模式
            model.eval()

            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)

            val_correct += torch.sum(pre_lab == b_y.data)

            val_num += b_x.size(0)

        # 计算并保存每一次迭代的 loss 值和准确率
        # 计算并保存训练集的 loss 值
        train_loss_all.append(train_loss / train_num)

        # 计算并保存训练集的准确率
        train_acc_all.append(train_correct.double().item() / train_num)

        # 计算并保存验证集的 loss 值
        val_loss_all.append(val_loss / val_num)

        # 计算并保存验证集的准确率
        val_acc_all.append(val_correct.double().item() / val_num)

        print("{} Train Loss:{:.4f} Train Accuracy: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{}   Val Loss:{:.4f}   Val Accuracy: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]

            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间为：{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))


    # 训练结束后，选择最优参数，保存最优参数模型
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "D:/PytorchProject/AlexNet/best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all,})

    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="Train Accuracy")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    AlexNet = AlexNet()

    train_dataloader, val_dataloader = train_val_data_process()

    train_process = train_model_process(AlexNet, train_dataloader, val_dataloader, num_epochs=25)

    matplot_acc_loss(train_process)



