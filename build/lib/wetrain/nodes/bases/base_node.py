import os
import sys

class node(object):

    '''
    该节点为所有节点的基类
    所有的数据节点和计算节点都需要继承自该类
    '''

    def __init__(self):

        self._node_attr = {'class':'base'}

        self.parent = [] #记录节点的父节点
        self.children = [] #表示节点的子节点

    def forward(self):
        pass

    def backward(self):
        pass
