import os
import torch
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import ResNet
from dataset import imageDataSet
from tqdm import tqdm
import dataset

NUM_CLASS = 176
BATCH_SIZE = 512
IMG_HEIGHT = 64
IMG_WIDTH = 64
N_FOLD = 5

saveFileName = 'data/submission.csv'
TEST_DATA_PATH = 'data/test.csv'
TRAIN_DATA_PATH = 'data/train.csv'


def test():
    model = []
    save_filename = []
    num_to_class = dataset.get_num2class()
    for i in range(N_FOLD):
        model.append(ResNet.resnext50_32x4d_init(num_class=NUM_CLASS, pretrained=True, finetune=True))
        save_filename.append('checkpoint_resnext50_' + str(i) + '.pth.tar')
        model[i].load_state_dict(torch.load(os.path.join('models', save_filename[i]))['state_dict'])
        model[i].eval()

    test_data_path = dataset.get_test_data().image.values

    test_data = imageDataSet(file_path=test_data_path, train=False, height=IMG_HEIGHT, width=IMG_WIDTH)
    test_data = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    predictions = []

    for batch in tqdm(test_data):
        images = batch
        predicts_set = []
        for i in range(N_FOLD):
            with torch.no_grad():
                predicts = model[i](images)
            predicts_set.append(predicts)

        predicts_mean = (sum(predicts_set) / len(predicts_set))
        predicts_mean = predicts_mean.argmax(dim=-1)
        predictions.extend(predicts_mean.data.numpy().tolist())

    predicts = []
    for i in tqdm(predictions):
        predicts.append(num_to_class[i])

    test_data = pd.read_csv(TEST_DATA_PATH)
    test_data['label'] = pd.Series(predicts)

    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Done!")


def valid():
    model = []
    save_filename = []
    for i in range(N_FOLD):
        model.append(ResNet.resnext50_32x4d_init(num_class=NUM_CLASS, pretrained=True, finetune=True))
        save_filename.append('checkpoint_resnext50_' + str(i) + '.pth.tar')
        model[i].load_state_dict(torch.load(os.path.join('models', save_filename[i]))['state_dict'])
        model[i].eval()

    valid_data = dataset.get_train_data()
    file_path = valid_data.image.values
    label = valid_data.label.values

    test_data = imageDataSet(file_path=file_path, label=label, height=IMG_HEIGHT, width=IMG_WIDTH)
    test_data = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    for batch in tqdm(test_data):
        images, labels = batch
        predictions = []
        predictions_rank = []
        for i in range(N_FOLD):
            with torch.no_grad():
                predicts = model[i](images)
            predictions.append(predicts)
            predictions_rank.append(F.softmax(predicts, dim=1))
            acc = (predicts.argmax(dim=-1) == labels).float().mean().data.numpy()

            print(f"Model_{i + 1:02d}: {acc}")

        com_mean = (sum(predictions) / len(predictions))
        rank_mean = sum(predictions_rank)

        print(f"Model_com_avg: {(com_mean.argmax(dim=-1) == labels).float().mean().data.numpy()}")
        print(f"Model_rank_avg: {(rank_mean.argmax(dim=-1) == labels).float().mean().data.numpy()}")


if __name__ == '__main__':
    valid()
    # test()

