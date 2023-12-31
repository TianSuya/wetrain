import math
import sys
import os
path = os.path.dirname(__file__)
sys.path.append(path)
from bases.cal_node import cal_node
from bases.data_node import data_node

class mul_node(cal_node):

    def __init__(self,data_1:data_node,data_2:data_node,data_3:data_node):

        super().__init__()

        self._node_attr['type'] = 'mul'

        self.data_1 = data_1 #设置当前节点需要的前后运算节点
        self.data_2 = data_2
        self.data_3 = data_3

        self.children.append(data_1) #加载到节点图中
        self.children.append(data_2)
        self.parent.append(data_3)

        self.data_1.parent.append(self)
        self.data_2.parent.append(self)
        self.data_3.children.append(self)

    def forward(self):
        #计算节点值
        self.data_3.data = self.data_1.data * self.data_2.data
        #设置节点状态
        self.is_forward = True
        self.is_backward = False
        pass

    def backward(self):
        #设置梯度
        if self.data_1.require_grad == True:
            self.data_1.grad += self.data_2.data * self.data_3.grad
            self.data_1._node_attr['backward_type'] = 'mul_backward'
        else:
            self.data_1.grad = 0.
            self.data_1._node_attr['backward_type'] = None

        if self.data_2.require_grad == True:
            self.data_2.grad += self.data_1.data * self.data_3.grad
            self.data_2._node_attr['backward_type'] = 'mul_backward'
        else:
            self.data_2.grad = 0.
            self.data_2._node_attr['backward_type'] = None

        #设置节点状态
        self.is_backward = True
        self.is_forward = False
        pass

if __name__ == '__main__':

    input1 = data_node().initialize(3)
    input2 = data_node().initialize(4)

    input3 = data_node()
    input3.grad = 0.4

    adder = mul_node(input1, input2, input3)
    adder.forward()

    print(input3.data)

    adder.backward()

    print(input1.grad)
    print(input2.grad)


