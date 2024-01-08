import os
import sys

import numpy as np

path = os.path.dirname(__file__)
sys.path.append(path)

class wtensor:

    def __init__(self,data = 0.,require_grad = True):

        self.parent = []  # 记录节点的父节点
        self.children = []  # 表示节点的子节点

        self._node_attr = {}
        self._node_attr['backward_type'] = None

        self.require_grad = require_grad
        self.data = np.array(data,dtype=np.float32)
        self.shape = np.shape(self.data)
        self.grad = np.zeros_like(self.data)
        self.dtype = 'float'
        self.is_forward = False
        self.is_backward = False


    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def set_value(self,data):
        self.data = np.array(data,dtype=np.float32)
        self.shape = np.shape(self.data)

    def clear_rela(self):
        self.parent = []
        self.children = []

    def forward(self):

        #等到所有的子节点都forward以后再执行
        for item in self.children:
            if item.is_forward == False:
                item.forward()

        self.is_forward =True
        self.is_backward = False

        for item in self.parent:
            if item.is_forward == False:
                item.forward()

        pass

    def backward(self):

        #等所有的父节点都反向传播以后再执行
        for item in self.parent:
            if item.is_backward == False:
                item.backward()

        if self.parent == []:
            self.grad = np.ones_like(self.data)

        self.is_backward = True
        self.is_forward = False

        for item in self.children:
            if item.is_backward == False:
                item.backward()

        return