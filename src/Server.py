'''
服务端实现
'''
from base.BaseServer import BaseServer
class Server(BaseServer):
    def __init__(self, model, testDataloader=None):
        super().__init__(model, testDataloader)