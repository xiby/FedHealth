'''
客户端基类
'''

class BaseClient():
    def __init__(self, model, optimizer, loss_fn, dataloader, testDataloader=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader
        self.testDataloader = testDataloader
    
    def loadParams(self, param):
        '''
        加载模型参数，子类来实现
        '''
        raise NotImplementedError

    def reportParams(self):
        '''
        上报参数，子类来实现
        '''
        raise NotImplementedError
    
    def trainModel(self):
        '''
        具体模型训练
        '''
        raise NotImplementedError
        
    def testModel(self):
        '''
        模型测试，子类来实现
        '''
        raise NotImplementedError