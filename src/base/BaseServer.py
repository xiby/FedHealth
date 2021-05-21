'''
服务端基类
'''

class BaseServer():
    def __init__(self, model, testDataloader=None):
        super().__init__()
        self.clients = []
        self.paramList = []
        self.param = None
        self.model = model
        self.testDataloader = testDataloader
    
    def addClient(self, client):
        self.clients.append(client)
    
    def aggregate(self):
        '''
        进行参数的聚合
        '''
        raise NotImplementedError

    def startTrainLoop(self, rounds):
        '''
        开启整个的训练
        '''
        for i in range(rounds):
            for client in self.clients:
                client.loadParams(self.param)
                client.trainModel()
                self.paramList.append(client.reportParams)
            self.aggregate()
    def testModel(self):
        '''
        对聚合后的模型进行训练
        '''
        raise NotImplementedError
            
