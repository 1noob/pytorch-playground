import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def resnet50_init(num_class=1000, pretrained=False, finetune=False):
    model = models.resnet50(pretrained=pretrained).to(device=device)
    num_fc = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=num_fc, out_features=num_class, bias=True),
    )

    if finetune:
        for parma in model.parameters():
            parma.requires_grad = False

        for parma in model.layer4.parameters():
            parma.requires_grad = True

        for parma in model.avgpool.parameters():
            parma.requires_grad = True

        for parma in model.fc.parameters():
            parma.requires_grad = True

    return model


def resnet18_init(num_class=1000, pretrained=False, finetune=False):
    model = models.resnet18(pretrained=pretrained).to(device=device)
    num_fc = model.fc.in_features

    if finetune:
        for parma in model.parameters():
            parma.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(in_features=num_fc, out_features=num_class, bias=True),
    )

    return model


def resnet101_init(num_class=1000, pretrained=False, finetune=False):
    model = models.resnet101(pretrained=pretrained).to(device=device)
    num_fc = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=num_fc, out_features=num_class, bias=True),
    )

    if finetune:
        for parma in model.parameters():
            parma.requires_grad = False

        for parma in model.layer3.parameters():
            parma.requires_grad = True

        for parma in model.layer4.parameters():
            parma.requires_grad = True

        for parma in model.avgpool.parameters():
            parma.requires_grad = True

        for parma in model.fc.parameters():
            parma.requires_grad = True

    return model

def resnext50_32x4d_init(num_class=1000, pretrained=False, finetune=False):
    model = models.resnext50_32x4d(pretrained=pretrained).to(device=device)
    num_fc = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=num_fc, out_features=num_class, bias=True),
    )

    if finetune:
        for parma in model.parameters():
            parma.requires_grad = False

        for parma in model.layer3.parameters():
            parma.requires_grad = True

        for parma in model.layer4.parameters():
            parma.requires_grad = True

        for parma in model.avgpool.parameters():
            parma.requires_grad = True

        for parma in model.fc.parameters():
            parma.requires_grad = True

    return model


if __name__ == '__main__':
    net = models.resnext50_32x4d()
    print(net)
