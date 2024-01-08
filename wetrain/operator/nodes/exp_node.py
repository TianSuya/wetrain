import sys
import os

import numpy as np

# path = os.path.dirname(__file__)
# sys.path.append(path)
from wetrain.operator.nodes.bases.cal_node import cal_node
from wetrain.wtensor.wtensor import wtensor

class exp_node(cal_node):

    def __init__(self,data_1:wtensor,data_2:wtensor,a:float):

        super().__init__()

        self._node_attr['type'] = 'exp'

        self.data_1 = data_1 #设置当前节点需要的前后运算节点
        self.data_2 = data_2
        self.a = float(a)

        self.children.append(data_1) #加载到节点图中
        self.parent.append(data_2)

        self.data_1.parent.append(self)
        self.data_2.children.append(self)

    def forward(self):

        # 如果还有子节点没有前向传播，则直接返回
        for item in self.children:
            if item.is_forward == False:
                item.forward()

        #计算节点值
        self.data_2.data = np.power(self.a,self.data_1.data)
        self.data_2.grad = np.zeros_like(self.data_2.data)
        #设置节点状态
        self.is_forward = True
        self.is_backward = False

        # 继续前向传播
        for item in self.parent:
            if item.is_forward == False:
                item.forward()

        return

    def backward(self):

        # 若还有父节点没有反向传播，则退出
        for item in self.parent:
            if item.is_backward == False:
                item.backward()

        #设置梯度
        self.data_1.grad += np.log(self.a) * np.power(self.a,self.data_1.data) * self.data_2.grad
        self.data_1._node_attr['backward_type'] = 'exp_backward'

        #设置节点状态
        self.is_backward = True
        self.is_forward = False

        for item in self.children:
            if item.is_backward == False:
                item.backward()

        return

if __name__ == '__main__':
    input1 = wtensor([[1,2],[3,4]])
    input2 = wtensor([[2,3],[3,4]])

    input2.grad = np.array([[0.1,0.2],[0.2,0.2]])

    adder = exp_node(input1, input2, 2)
    adder.forward()

    print(input2.data)

    adder.backward()

    print(input1.grad)
    print(input2.grad)