import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F  # 包含一些没有参数的神经网络函数，例如激活函数和 dropout

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 定义 AlexNet 的结构
        self.ReLu = nn.ReLU()  # 激活函数
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)  # 卷积层
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2) # 最大池化层
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)  # 卷积层
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2) # 最大池化层
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 展平
        self.flatten = nn.Flatten() # 将池化后的三维特征图展平为一维向量，供全连接层使用

        self.f1 = nn.Linear(in_features=256*6*6, out_features=4096)  # 全连接层
        self.f2 = nn.Linear(in_features=4096, out_features=4096)  # 全连接层

        # out_features=10 表示分类任务中的 10 个类别
        self.f3 = nn.Linear(in_features=4096, out_features=10)  # 全连接层


    def forward(self, x):
        x = self.ReLu(self.c1(x))
        x = self.s2(x)
        x = self.ReLu(self.c3(x))
        x = self.s4(x)
        x = self.ReLu(self.c5(x))
        x = self.ReLu(self.c6(x))
        x = self.ReLu(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)

        x = self.ReLu(self.f1(x))
        # 在全连接层之后加入 dropout，随机丢弃一部分神经元，防止过拟合
        x = F.dropout(x, p=0.5)
        x = self.ReLu(self.f2(x))
        x = F.dropout(x, p=0.5)
        x = self.f3(x)

        return x

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlexNet().to(device)

    print(summary(model, (1, 227, 227)))