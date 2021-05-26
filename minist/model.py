import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积(Conv2d)->激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 输入图像大小(1,28,28)
            nn.Conv2d(
                in_channels=1,      # 数据集是灰度图像只有一个通道
                out_channels=16,    # n_filters 卷积核的通道
                kernel_size=5,      # filter_size = 5x5
                stride=1,           # 步长
                padding=2,          # 想要conv2d输出的图片长宽不变就进行补零操作 padding = floor(kernel_size/2)
            ),
            # 输出图像大小(16,28,28)
            nn.ReLU(),
            # 在2x2空间下采样
            nn.MaxPool2d(kernel_size=2),
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小 (16,14,14)
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 把每一个批次的每一个输入都拉成一个维度 即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w] 所以x.size(0)就是batch_size
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
