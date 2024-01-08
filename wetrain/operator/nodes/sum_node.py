import sys
import os

import numpy as np

# path = os.path.dirname(__file__)
# sys.path.append(path)
from wetrain.operator.nodes.bases.cal_node import cal_node
from wetrain.wtensor.wtensor import wtensor

class sum_node(cal_node):

    def __init__(self,data_in:wtensor,data_out:wtensor,dim:int):

        super().__init__()

        self._node_attr['type'] = 'sum'

        self.dim = dim
        self.data_in = data_in
        self.data_out = data_out

        self.children.append(data_in)
        data_in.parent.append(self)

        self.parent.append(data_out)
        self.data_out.children.append(self)


    def forward(self):

        # 如果还有子节点没有前向传播，则直接返回
        for item in self.children:
            if item.is_forward == False:
                item.forward()

        #计算节点值
        self.data_out.data = np.sum(self.data_in.data,axis=self.dim)
        self.data_out.grad = np.zeros_like(self.data_out.data)
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
        self.data_in.grad += np.zeros_like(self.data_in.data) + np.expand_dims(self.data_out.grad,axis=self.dim)
        self.data_in._node_attr['backward_type'] = 'sum_backward'

        #设置节点状态
        self.is_backward = True
        self.is_forward = False

        for item in self.children:
            if item.is_backward == False:
                item.backward()

        return

if __name__ == '__main__':

    input1 = wtensor([[1,2,3],[4,5,6]])
    input2 = wtensor()
    input2.grad = [0.3,0.2,0.4]

    adder = sum_node(input1,input2,0)
    input1.is_forward = True
    adder.forward()

    print(input2.data)

    input2.is_backward = True
    adder.backward()

    print(input1.grad)



