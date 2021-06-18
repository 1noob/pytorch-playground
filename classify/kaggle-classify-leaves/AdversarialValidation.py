import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch
import torch.nn as nn
import dataset
import ResNet
from tqdm import tqdm
from dataset import imageDataSet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

NUM_CLASS = 2
EPOCH = 20
LR = 1e-3
BATCH_SIZE = 64
WARMUP_STEP = 5
IMG_HEIGHT = 64
IMG_WIDTH = 64
N_FOLD = 5
np.random.seed(42)
train = dataset.get_train_data()
test = dataset.get_test_data()

model = ResNet.resnet18_init(num_class=NUM_CLASS, pretrained=True, finetune=True)
optimizer = torch.optim.Adam(model.parameters(), LR)
loss_func = nn.CrossEntropyLoss()


train['TARGET'] = 1
test['TARGET'] = 0

data_df = pd.concat(( train, test )).reset_index().drop(['index'], axis=1)
x = data_df.drop( [ 'TARGET', 'label' ], axis=1 )
y = data_df.TARGET

splitter = StratifiedKFold(n_splits=N_FOLD, shuffle=True)

for fold, (trn_idx, val_idx) in enumerate(splitter.split(x, y)):
    val_aucs = []
    train_data = imageDataSet(file_path=x.image.values[trn_idx], label=y.values[trn_idx], mode='train', height=IMG_HEIGHT, width=IMG_WIDTH)
    train_data = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model.train()
    for batch in tqdm(train_data):
        images, labels = batch
        predicts = model(images)

        loss = loss_func(predicts, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    valid_data = imageDataSet(file_path=x.image.values[val_idx], label=y.values[val_idx],
                                 mode='valid', height=IMG_HEIGHT, width=IMG_WIDTH)
    valid_data = data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for batch in tqdm(valid_data):
        images, labels = batch
        with torch.no_grad():
            predicts = model(images)
        loss = loss_func(predicts, labels)

        val_score = roc_auc_score(labels.data.numpy(), predicts.argmax(dim=-1).data.numpy())
        val_aucs.append(val_score)

    score = sum(val_aucs) / val_aucs.__len__()
    print(f"[ Valid | {fold + 1:02d}/{N_FOLD:02d} ] val_auc: {score:.5f}")




