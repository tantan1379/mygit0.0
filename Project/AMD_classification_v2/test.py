import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed

# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# 3. test model on public dataset and save the probability matrix
def test(test_loader, model, folds):
    model.cuda()
    model.eval()
    for _, (input, target) in enumerate(tqdm(test_loader)):
        input = input.cuda()
        target = torch.from_numpy(np.array(target)).long().cuda()  # 范式
        output = model(input)  # 将input输入模型得到预测输出
        precision1, _ = accuracy(output, target, topk=(1, 2))
    return precision1[0]


def evaluate(md, loader):
    correct_num = 0
    target_num = torch.zeros((1, 3))
    predict_num = torch.zeros((1, 3))
    acc_num = torch.zeros((1, 3))
    total_num = len(loader.dataset)

    for _, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        output = md(data)
        _, prediction = torch.max(output, 1)
        pre_mask = torch.zeros(output.size()).scatter_(1, prediction.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)

    accuracy = acc_num.sum(1) / target_num.sum(1)
    accuracy = (accuracy.numpy()[0] * 100).round(3)

    return accuracy

# 4. more details to build main function
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep + str(fold) + os.sep)
    # 4.2 get model and optimizer
    model = get_net()
    # model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()
    # 4.5 get files and split for K-fold dataset
    test_files = get_files(config.test_data)
    # 4.5.4 load dataset
    test_dataloader = DataLoader(ChaojieDataset(test_files), batch_size=1, shuffle=False, pin_memory=False)
    best_model = torch.load("checkpoints/best_model/%s/0/model_best.pth.tar" % config.model_name)
    model.load_state_dict(best_model["state_dict"])
    precision = evaluate(model,test_dataloader)
    print(precision)


if __name__ == "__main__":
    main()
