import os
import sys

path = os.path.dirname(__file__)
sys.path.append(path)


from base_node import node

class cal_node(node):

    def __init__(self):
        super().__init__()

        self._node_attr['class'] = 'cal'

        self.name = 'base_cal'
        self.require_grad = False
        self.is_backward = False
        self.is_forward = False

    def zero_grad(self):
        pass

    def backward(self):
        pass

    def forward(self):
        pass

