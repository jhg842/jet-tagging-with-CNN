import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import namedtuple
from pytorchtools import EarlyStopping
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from model.GRU import GRU
from model.ResNet import ResNet, BasicBlock
from model.CNN import CNN
# from model.CNN_residual import CNN_connection, Residual
# from models.XAI import CNN

from eval_performance import eval_jets
from image_making.image_processing import processing

ResNetConfig = namedtuple('ResNetConfig',['block','n_blocks','channels'])
residual_config = namedtuple('residual_config',['block','channels'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_argments():
    parser = argparse.ArgumentParser(description="plotting of KEK beam test data taking", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", "-g", type=str)
    parser.add_argument("--epochs", "-e", type=int, default='50', help='set to number of epochs')
    parser.add_argument("--lr", "-l", type=float, default='0.001', help='set to learning rate')
    parser.add_argument("--batch", "-b", type=int, default='1024', help='set to batch size')
    parser.add_argument("--model","-m", type=str, default=None, help="set to NN model")
    
    args=parser.parse_args()
    return args

def training(epoch, model,loss_fn,schedular,optimizer, trainloader, validloader):
    correct = 0
    total = 0
    loss = 0

    model.train()

    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)

        # y_pred = model(x)[0]
        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        with torch.no_grad():
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
            loss += loss.item()

    epoch_loss = loss / len(trainloader)
    epoch_acc = correct / total

    valid_correct = 0
    valid_total = 0
    valid_loss = 0

    model.eval()
    with torch.no_grad():
        for vx,vy in validloader:
            vx = vx.to(device)
            vy = vy.to(device)
            # y_pred = model(vx)[0]
            y_pred = model(vx)
            loss = loss_fn(y_pred,vy)
            y_predict = torch.argmax(y_pred, dim = 1)
            valid_correct += (y_predict == vy).sum().item()
            valid_total += vy.size(0)
            valid_loss += loss.item()

    epoch_valid_loss = valid_loss / len(validloader)
    epoch_valid_acc = valid_correct / valid_total

    schedular.step()        

    print('epoch:',epoch,
    'loss:', round(epoch_loss.item(),3),
    'accuracy:', round(epoch_acc,3),
    'valid_loss:', round(epoch_valid_loss,3),
    'valid_accuracy:', round(epoch_valid_acc,3))

    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc


def main():
    start_time = datetime.now()

    seed = 777
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = get_argments()

    # if config.model == 'residual':
    #     CNN_residual = residual_config(Residual, channels=[64,128,256,512])
    #     model = CNN_connection(CNN_residual, 512).to(device)

    if config.model == 'resnet':
        resnet18_config = ResNetConfig(block=BasicBlock, n_blocks=[2,2,2,2], channels=[64,128,256,512])
        model = ResNet(resnet18_config,3).to(device)

    elif config.model == 'cnn':
        model = CNN(3,[64,128,256,512],512,3, True).to(device)

    print('Model:', model)

    # datasets
    trainloader = processing(config.batch, True, 'train',4)
    validloader = processing(config.batch, False, 'validation',4)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    # schedular = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20,70], gamma=0.5)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones= [5, 15], gamma= 0.1)

    tr_loss = []; tr_acc = []; val_loss = []; val_acc = []; total_epochs = []

    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Training and validation process
    for epoch in tqdm(range(config.epochs), desc='Training Progress'):
        train_loss, train_acc, valid_loss, valid_acc = training(epoch, model,loss_fn,schedular, optimizer, trainloader, validloader)

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            break

        tr_loss.append(train_loss.item())
        tr_acc.append(train_acc)
        val_loss.append(valid_loss)
        val_acc.append(valid_acc)
        total_epochs.append(epoch)

    torch.save(model.state_dict(), f'/home/jhg842/jet_tagging/eval_performance/saved_model/{config.model}_full_ITS3_20k_dot_5060.pth')
    
    # save the train, valid plots
    eval_jets.loss(total_epochs, tr_loss, val_loss)
    eval_jets.accuracy(total_epochs, tr_acc, val_acc)


if __name__ == "__main__":
    main()
