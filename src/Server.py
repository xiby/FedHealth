'''
服务端实现
'''
import torch

from base.BaseServer import BaseServer
class Server(BaseServer):
    def __init__(self, model, dataloader, optimizer, loss_fn, testDataloader=None):
        super().__init__(model, testDataloader)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
    def trainModel(self):
        size = len(self.dataloader.dataset)
        for batch, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"server batch: {batch:>5d}  loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
    def aggregate(self):
        '''
        TODO server update its model by aligning with user model
        '''
        pass
    def testModel(self):
        '''
        全局模型不需要测试
        '''
        pass
    def startTrainLoop(self, rounds=80):
        '''
        开启训练循环
        默认训练80轮
        '''
        self.trainModel()
        for client in self.clients:
            client.loadParams(self.model.state_dict())
            client.trainModel()
            self.paramList.append(client.reportParams())
        self.aggregate()
        
        


