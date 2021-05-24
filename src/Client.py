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
    
    def _freeze_model(self):
        '''
        将前几层的梯度设置为不更新
        '''
        for para in self.model.conv1.parameters():
            para.requires_grad = False
        for para in self.model.mp1.parameters():
            para.requires_grad = False
        for para in self.model.conv2.parameters():
            para.requires_grad = False
        for para in self.model.mp2.parameters():
            para.requires_grad = False
    def _unfreeze_model(self):
        for para in self.model.conv1.parameters():
            para.requires_grad = True
        for para in self.model.mp1.parameters():
            para.requires_grad = True
        for para in self.model.conv2.parameters():
            para.requires_grad = True
        for para in self.model.mp2.parameters():
            para.requires_grad = True
    def trainModel(self, state):
        '''
        TODO 论文中对于客户端有两种训练方法，一种是一开始利用自身数据进行训练，第二种是训练迁移后的训练，需要思考怎么实现
        利用state来控制训练过程，当state为0时，利用自身进行训练，此时更新前面几层的参数，当state为1时，进行迁移后的训练，此时不更新前面几层的参数
        '''
        if state != 0 and state != 1:
            raise Exception("illegal parameter of state")
        elif state == 0:
            self._unfreeze_model()
            pass
        else:
            self._freeze_model()
        pass

    def testModel(self):
        pass
    