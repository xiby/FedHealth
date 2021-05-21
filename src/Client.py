'''
客户端具体实现
'''
from base.BaseClient import BaseClient

class Client(BaseClient):
    def __init__(self, model, optimizer, loss_fn, dataloader, testDataloader=None):
        super().__init__(model, optimizer, loss_fn, dataloader, testDataloader)