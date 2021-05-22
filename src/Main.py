'''
运行主文件
'''
from FederatedDataset import FederatedDataset

import torch.nn as nn
from torch.utils.data import DataLoader
if __name__ == "__main__":
    dataset = FederatedDataset(root='../datas/UCI_DATASET', subjects=[26])
    dataloader = DataLoader(dataset, batch_size=64)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to('cpu')
        # print(X.shape)
        X = X.unsqueeze(1)
        conv1 = nn.Conv1d(1, 1, 9)
        mp1 = nn.MaxPool1d(3)
        out = conv1(X)
        out = mp1(out)
        out = conv1(out)
        out = mp1(out)

        fc1 = nn.Linear(58, 27)
        out = fc1(out)
        fc2 = nn.Linear(27, 6)
        out = fc2(out)
        out = out.squeeze(1)
        sm = nn.Softmax(dim=1)
        out = sm(out)
        print(out.argmax(1))
        print(y)
    print("Main")