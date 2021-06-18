import PIL
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import AutoAugmentPolicy
from tqdm import tqdm
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RepeatedKFold


TEST_DATA_PATH = 'data/test.csv'
TRAIN_DATA_PATH = 'data/train.csv'

labels = pd.read_csv(TRAIN_DATA_PATH)['label'].values
label_list = sorted(list(set(labels)))
n_classes = len(label_list)

class_to_num = dict(zip(label_list, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}

def get_num2class():
    return num_to_class


def get_train_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    file = train_df['image'].values
    label = train_df['label'].values
    train_df['image'] = [os.path.join("data", i) for i in file]
    train_df['label'] = [class_to_num[i] for i in label]

    return train_df


def get_test_data():
    test_df = pd.read_csv(TEST_DATA_PATH)
    file = test_df['image'].values
    test_df['image'] = [os.path.join("data", i) for i in file]

    return test_df


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

AutoAug_preprocess = transforms.Compose([
    # 这玩意自带维度转换功能
    transforms.AutoAugment(AutoAugmentPolicy.SVHN),
    transforms.ToTensor(),
])

SimpleAug_preprocess = transforms.Compose([
    # 这玩意自带维度转换功能
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(2),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


def AutoAugment_loader(path, size):
    img = Image.open(path)
    img = AutoAug_preprocess(img)
    img = transforms.Resize(size=size)(img)
    return normalize(img)


def SimpleAug_loader(path, size):
    img = Image.open(path)
    img = SimpleAug_preprocess(img)
    img = transforms.Resize(size=size)(img)
    return normalize(img)


def common_loader(path, size):
    img = Image.open(path)
    img = transforms.ToTensor()(img)
    img = transforms.Resize(size=size)(img)
    return normalize(img)


class imageDataSet(Dataset):
    def __init__(self, file_path, label=None, train=True, tta=True, auto_aug=False, size=64):
        if train or tta:
            if auto_aug:
                self.loader = AutoAugment_loader
            else:
                self.loader = SimpleAug_loader
        else:
            self.loader = common_loader
        self.images = file_path
        self.labels = label
        self.size = size
        self.train = train

    def __getitem__(self, index):
        fn = self.images[index]
        image = self.loader(fn, self.size)

        if self.train:
            label = self.labels[index]
            return image, label

        return image

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    print(get_train_data())
