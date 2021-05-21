'''
联邦下的数据集
'''
import random

from torch.utils.data import Dataset

class FederatedDataset(Dataset):
    # TODO 添加实现
    def __init__(self, source, size=600, transform=None):
        pass

    def __getitem__(self, index):
        return None
    
    def __len__(self):
        return 0