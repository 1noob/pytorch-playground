import argparse
import torch
import os
import torch.utils.data as data
import ResNet
from tqdm import tqdm
from dataset import imageDataSet
import dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.focal_loss import MultiFocalLoss
from sklearn.model_selection import RepeatedKFold, StratifiedKFold

# NUM_CLASS = 176
# EPOCH = 50
# LR = 0.02
# BATCH_SIZE = 64
# WARMUP_STEP = 10
# IMG_SIZE = 64
# N_FOLD = 5


parser = argparse.ArgumentParser(description='PyTorch Classify-leaves Training')

parser.add_argument('--model', default='resnext50', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--path', default='models', type=str,
                    help='Path of model to save')
parser.add_argument('--checkpoint', default='checkpoint.pth.tar', type=str,
                    help='checkpoint name of model to save')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--classes', type=int, default=176, metavar='NUM_CLASS',
                    help='number of label classes (Model default if None)')
parser.add_argument('--size', type=int, default=64, metavar='IMG_SIZE',
                    help='Image patch size (default: 64)')
parser.add_argument('-b', '--batch', type=int, default=64, metavar='BATCH_SIZE',
                    help='input batch size for training (default: 64)')
parser.add_argument('--fold', type=int, default=5, metavar='N_Fold',
                    help='input batch size for training (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='input batch size for training (default: 0.01)')
parser.add_argument('--epoch', type=int, default=50, metavar='EPOCH',
                    help='EPOCH for training (default: 50)')
parser.add_argument('--warmup', type=int, default=10, metavar='WARMUP_STEP',
                    help='learning rate warmup for training (default: 10)')
parser.add_argument('--aug', action='store_true', default=False,
                    help='Use AutoAugment for image (if avail)')
parser.add_argument('--finetune', action='store_true', default=False,
                    help='Use finetune version of specified network (if avail)')
parser.add_argument('--loadstate', action='store_true', default=False,
                    help='continue train')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
splitter = StratifiedKFold(n_splits=args.fold, shuffle=True)
data_df = dataset.get_train_data()


def stacking(model, is_load=False, model_name=None):
    if is_load:
        for i in range(args.fold):
            model[i].load_state_dict(torch.load(os.path.join(args.path, model_name[i]))['state_dict'])

    best_acc = 0
    start_epoch = 0
    optimizer = []
    loss_func = []
    scheduler_steplr = []
    scheduler_warmup = []
    for i in range(args.fold):
        optimizer.append(torch.optim.Adam(model[i].parameters(), args.lr))
        loss_func.append(MultiFocalLoss(num_class=args.classes, alpha=0.2))

        if is_load:
            checkpoint = torch.load(os.path.join(args.path, model_name[i]))
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            optimizer[i].load_state_dict(checkpoint['optimizer'])

        scheduler_steplr.append(CosineAnnealingLR(optimizer[i], T_max=args.epoch - args.warmup, eta_min=3e-4))
        scheduler_warmup.append(GradualWarmupScheduler(optimizer[i],
                                                       multiplier=1,
                                                       total_epoch=args.warmup,
                                                       after_scheduler=scheduler_steplr[i]))

        optimizer[i].zero_grad()
        optimizer[i].step()
        scheduler_warmup[i].step()

    for epoch in range(start_epoch, args.epoch):

        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []

        for i, (trn_idx, val_idx) in enumerate(splitter.split(data_df.image, data_df.label)):
            model[i].train()
            train_data = imageDataSet(file_path=data_df.image.values[trn_idx], label=data_df.label.values[trn_idx],
                                      size=args.size, auto_aug=args.aug)
            train_data = data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
            tmp_acc = []
            tmp_loss = []
            with tqdm(train_data) as batchs:
                for batch in batchs:
                    images, labels = batch
                    predicts = model[i](images)
                    loss = loss_func[i](predicts, labels)
                    optimizer[i].zero_grad()
                    loss.backward()
                    optimizer[i].step()

                    tmp_acc.append((predicts.argmax(dim=-1) == labels).float().mean())
                    tmp_loss.append(loss.data.numpy())

            train_loss.append(sum(tmp_loss) / len(tmp_loss))
            train_acc.append(sum(tmp_acc) / len(tmp_acc))
            print(f"[ Train | Model{i + 1:02d}/{args.fold:02d} ] loss = {train_loss[i]:.5f} | acc = {train_acc[i]:.5f}")

            model[i].eval()
            valid_data = imageDataSet(file_path=data_df.image.values[val_idx], label=data_df.label.values[val_idx],
                                      size=args.size, auto_aug=args.aug)
            valid_data = data.DataLoader(valid_data, batch_size=args.batch, shuffle=True)
            tmp_acc = []
            tmp_loss = []
            with tqdm(valid_data) as batchs:
                for batch in batchs:
                    images, labels = batch
                    with torch.no_grad():
                        predicts = model[i](images)
                    loss = loss_func[i](predicts, labels)

                    tmp_acc.append((predicts.argmax(dim=-1) == labels).float().mean())
                    tmp_loss.append(loss.data.numpy())

            valid_loss.append(sum(tmp_loss) / len(tmp_loss))
            valid_acc.append(sum(tmp_acc) / len(tmp_acc))
            print(f"[ Valid | Model{i + 1:02d}/{args.fold:02d} ] loss = {valid_loss[i]:.5f} | acc = {valid_acc[i]:.5f}")

            scheduler_warmup[i].step()

        print(f"[ Epoch | {epoch + 1:02d}/{args.epoch:02d} ] lr = {optimizer[0].param_groups[0]['lr']:.5f} |"
              f" trn_loss = {sum(train_loss) / len(train_loss):.5f} |"
              f" trn_acc = {sum(train_acc) / len(train_acc):.5f} |"
              f" val_loss = {sum(valid_loss) / len(valid_loss):.5f} |"
              f" val_acc = {sum(valid_acc) / len(valid_acc):.5f}")

        epoch_acc = sum(valid_acc) / len(valid_acc)
        if epoch_acc > best_acc:
            for i in range(args.fold):
                best_acc = epoch_acc
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model[i].state_dict(),
                    'optimizer': optimizer[i].state_dict(),
                    'best_acc': best_acc,
                }
                torch.save(state, os.path.join(args.path, model_name[i]))
            print('saving model with acc {:.3f}'.format(best_acc))


def main():
    model = []
    model_name = []
    for i in range(args.fold):
        model.append(ResNet.resnext50_32x4d_init(num_class=args.classes, pretrained=args.pretrained, finetune=args.finetune))
        model_name.append(f'{i+1:02d}_{args.checkpoint}')

    stacking(model=model, is_load=args.loadstate, model_name=model_name)


# python train.py --finetune --pretrained --aug --checkpoint 'resnext_SVHM.pth.tar'


if __name__ == '__main__':
    main()

