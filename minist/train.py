import torch
import torch.nn as nn
import torchvision
import cv2
from model import CNN
import dataset


EPOCH = 3           # 训练整批数据的次数
LR = 0.001          # 学习率
BATCH_SIZE = 50     # 批训练大小

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同
train_data, test_data = dataset.get_data(batch_size=BATCH_SIZE)


def train(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_data):      # 分配batch data
            predicts = cnn(b_x)                               # 先将数据放到cnn中计算output
            loss = loss_func(predicts, b_y)                   # 输出和真实标签的loss
            optimizer.zero_grad()                             # 清除之前学到的梯度的参数
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                predicts = cnn(test_data.data)
                predicts = torch.max(predicts, 1)[1].data.numpy()
                accuracy = float((predicts == test_data.targets.data.numpy()).astype(int)
                                 .sum()) / float(test_data.targets.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

    torch.save(cnn.state_dict(), 'model/minist.pkl')  # 保存模型


def show(cnn):
    cnn.load_state_dict(torch.load('model/minist.pkl'))
    inputs = test_data.data[:32]
    test_output = cnn(inputs)
    predicts = torch.max(test_output, 1)[1].data.numpy()
    print(predicts, 'prediction number')  # 打印识别后的数字

    img = torchvision.utils.make_grid(inputs)
    img = img.numpy().transpose(1, 2, 0)

    cv2.imshow('win', img)  # opencv显示需要识别的数据图片
    cv2.waitKey(0)


def main():
    cnn = CNN()
    train(cnn)
    show(cnn)


if __name__ == '__main__':
    main()
