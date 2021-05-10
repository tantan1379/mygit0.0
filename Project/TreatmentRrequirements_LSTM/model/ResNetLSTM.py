import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torchvision as tv
import time
import os
from torch.utils.data import DataLoader
from config import config


class ResNetLSTM(nn.Module):
    def __init__(self,lstm_hidden_size=2000,lr=config.lr):
        features = 18432
        layers = 2
        output = 1
        self.resnet = tv.models.resnet50(pretrained=True)
        self.final_pool = torch.nn.MaxPool2d(3, 2)
        self.LSTM = nn.LSTM(features,hidden,layers,batch_first=True)
        self.Linear = nn.Linear(hidden,output)

    def forward(self,x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.final_pool(x)
        x = x.flatten(start_dim=1)


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from dataloader.dataset import TreatmentRrequirement
    train_dataset = TreatmentRrequirement(config.data_path,transform='train')
    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,num_workers=0)
    for i,(x,y) in enumerate(train_loader):
        print(i)
        print(x.size())
        print(y.size())
        print("")