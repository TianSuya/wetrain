import math
import sys
import os
path = os.path.dirname(__file__)
sys.path.append(path)
from bases.cal_node import cal_node
from bases.data_node import data_node

class pow_node(cal_node):

    def __init__(self,data_1:data_node,data_2:data_node,n:float):

        super().__init__()

        self._node_attr['type'] = 'pow'

        self.data_1 = data_1 #设置当前节点需要的前后运算节点
        self.data_2 = data_2
        self.n = float(n) #幂指数

        self.children.append(data_1) #加载到节点图中
        self.parent.append(data_2)

        self.data_1.parent.append(self)
        self.data_2.children.append(self)

    def forward(self):
        try:
            #计算节点值
            self.data_2.data = math.pow(self.data_1.data,self.n)
            #设置节点状态
            self.is_forward = True
            self.is_backward = False
        except:
            print(self.data_1.data)
        pass

    def backward(self):
        #设置梯度
        if self.data_1.require_grad == True:
            self.data_1.grad += self.n * math.pow(self.data_1.data,self.n-1) * self.data_2.grad
            self.data_1._node_attr['backward_type'] = 'pow_backward'
        else:
            self.data_1.grad = 0.
            self.data_1._node_attr['backward_type'] = None

        #设置节点状态
        self.is_backward = True
        self.is_forward = False
        pass

if __name__ == '__main__':
    input1 = data_node(3)
    input2 = data_node()
    input2.grad = 0.4

    adder = pow_node(input1, input2, 2)
    adder.forward()

    print(input2.data)

    adder.backward()

    print(input1.grad)
    print(input2.grad)