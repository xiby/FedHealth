import torch.nn as nn
class Model(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        # self.conv1 = nn.Conv1d()
        self.conv1 = nn.Conv1d(1, 1, 9)
        self.mp1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv2d(1, 1, 9)
        self.mp2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(58, 27)
        self.fc2 = nn.Linear(27, 6)
        self.sm = nn.Softmax(dim=1)
        self.frozen = frozen
    
    def forward(self, X):
        out = X.unsqueeze(1)
        out = self.conv1(out)
        out = self.mp1(out)
        out = self.conv2(out)
        out = self.mp2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.squeeze(1)
        out = self.sm(out)
        return out
        
