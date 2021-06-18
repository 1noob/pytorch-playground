import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision

DOWNLOAD_MNIST = False


def get_data(batch_size, data_root='./data'):
    train_data = torchvision.datasets.MNIST(
        root=data_root,  # 保存或提取的位置
        train=True,      # true表明数据用于训练
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,
    )

    test_data = torchvision.datasets.MNIST(
        root=data_root,
        train=False,
        transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,
    )

    test_data.data = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)

    # 批训练50个samples, 1 channel, 28x28 (50,1,28,28)
    # Torch中的DataLoader是用来包装数据的工具, 它能帮我们有效迭代数据, 这样就可以进行批训练
    train_data = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True  # 打乱数据
    )
    return train_data, test_data
