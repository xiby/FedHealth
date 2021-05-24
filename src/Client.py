'''
客户端具体实现
'''
from base.BaseClient import BaseClient

class Client(BaseClient):
    def __init__(self, model, optimizer, loss_fn, dataloader, testDataloader=None):
        super().__init__(model, optimizer, loss_fn, dataloader, testDataloader)
    
    def loadParams(self, param):
        self.model.load_state_dict(param)

    def reportParams(self):
        return self.model.state_dict()
    
    def trainModel(self):
        '''
        TODO 论文中对于客户端有两种训练方法，一种是一开始利用自身数据进行训练，第二种是训练迁移后的训练，需要思考怎么实现
        '''
        pass

    def testModel(self):
        pass
    