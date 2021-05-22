'''
运行主文件
'''
from FederatedDataset import FederatedDataset

import torch.nn as nn

from torch.utils.data import DataLoader
from Model import Model

def getClientFrozenModel():
    model = Model()
    for para in model.conv1.parameters():
        para.requires_grad = False
    for para in model.mp1.parameters():
        para.requires_grad = False
    for para in model.conv2.parameters():
        para.requires_grad = False
    for para in model.mp2.parameters():
        para.requires_grad = False
    return model
if __name__ == "__main__":
    print("Main starts")