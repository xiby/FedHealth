'''
联邦下的数据集
'''
import random
import torch

from torch.utils.data import Dataset

class FederatedDataset(Dataset):
    def __init__(self, root, size=600, transform=None,  subjects=None, train=True):
        self.dataDir = root
        suffix = ''
        if train:
            self.dataDir += ('/train/')
            suffix = '_train.txt'
        else:
            self.dataDir += '/test/'
            suffix = '_test.txt'
        self.datas = []
        self.labels = []
        self.size = 0
        with open(self.dataDir + 'X' + suffix) as dataFile, open(self.dataDir + 'y' + suffix) as labelFile, open(self.dataDir + 'subject' + suffix) as subjectFile:
            for subject in subjectFile:
                data = dataFile.readline()
                label = labelFile.readline()
                if subjects is None or int(subject) in subjects:
                    self.datas.append(self._process_line(data))
                    self.labels.append(int(label) - 1)
                    self.size += 1
                else:
                    continue
    def _process_line(self, line):
        l = line.strip().split(" ")
        trueL = []
        for data in l:
            if len(data) > 0:
                trueL.append(float(data))
        return torch.FloatTensor(trueL)
    def __getitem__(self, index):
        return self.datas[index], self.labels[index]
    
    def __len__(self):
        return self.size

if __name__ == "__main__":
    dataset_full = FederatedDataset('../datas/UCI_DATASET/')
    dataset_26 = FederatedDataset('../datas/UCI_DATASET/', subjects=[26])
    print(len(dataset_full))
    print(len(dataset_26))