import os
import sys
path = os.path.dirname(__file__)
sys.path.append(path)
import base_node
class data_node(base_node.node):

    def __init__(self,data = 0.,require_grad = True):
        super().__init__()

        self._node_attr['class'] = 'data'
        self._node_attr['backward_type'] = None

        self.require_grad = require_grad
        self.grad = 0.
        self.dtype = 'float'
        self.data = float(data)

    def zero_grad(self):
        self.grad = 0.

    def set_value(self,data:float):
        self.data = float(data)

    def initialize(self,number:float):
        self.data = float(number)
        '''
        用于初始化节点的值
        '''
        return self

    def forward(self):
        pass

    def backward(self):
        pass