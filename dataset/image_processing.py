from posixpath import split
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
import splitfolders
################################################################################################ library 
# splitfolders.ratio('/home/jhg842/jet_tagging/data/dot/80100',output = '/home/jhg842/jet_tagging/data/dot/80100', seed = 1337, ratio=(0.6,0.4))
################################################################################################data split
def processing(batchsize=32,shuffles = True, option=None,workers=4):
    trans = transforms.Compose([transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]) # normalize((mean),(std)) gray:1, color:3


    trainset = torchvision.datasets.ImageFolder(root='/home/jhg842/jet_tagging/data/line/4050/test', transform=trans)

    valset = torchvision.datasets.ImageFolder(root='/home/jhg842/jet_tagging/data/line/4050/test', transform = trans)

    testset =  torchvision.datasets.ImageFolder(root = '/home/jhg842/jet_tagging/data/dot/6080/test', transform = trans)

################################################################################################ data preprocessing
    
    trainloader = DataLoader(trainset, batch_size = batchsize, shuffle=shuffles, num_workers = workers)
    valloader = DataLoader(valset, batch_size= batchsize, shuffle=False, num_workers=workers)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=shuffles,num_workers=workers)

    if option == 'train':
        return trainloader
    elif option == 'validation':
        return valloader
    elif option == 'test':
        return testloader
    else:
        return None
    

    
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(labels) #데이터 꺼내보는 용도

############################################################################################### data load

# def rnn_process(batchsize,option = None, workers = 4, trainingset,validset, testset):
#     train_loader = DataLoader(trainingset, batch_size = batchsize, shuffle=True, num_workers=workers)
#     valid_loader = DataLoader(validset, batch_size = batchsize, shuffle=False, num_workers=workers)
#     test_loader = DataLoader(testset, batch_size = batchsize, shuffle=False, num_workers=workers)

#     if option =='train':
#         return train_loader
#     elif option =='valid':
#         return valid_loader
#     elif option =='test':
#         return test_loader