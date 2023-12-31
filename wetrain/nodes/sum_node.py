import sys
import os
path = os.path.dirname(__file__)
sys.path.append(path)
from bases.cal_node import cal_node
from bases.data_node import data_node

class sum_node(cal_node):

    def __init__(self,data_in:list,data_out:data_node):

        super().__init__()

        self._node_attr['type'] = 'add'

        self.data_in = data_in
        self.data_out = data_out

        for item in self.data_in:
            self.children.append(item)
            item.parent.append(self)

        self.parent.append(data_out)
        self.data_out.children.append(self)


    def forward(self):
        #计算节点值
        self.data_out.data = 0.
        for item in self.data_in:
            self.data_out.data += item.data
        #设置节点状态
        self.is_forward = True
        self.is_backward = False
        pass

    def backward(self):
        #设置梯度
        for item in self.data_in:
            if item.require_grad == True:
                item.grad += 1. * self.data_out.grad
                item._node_attr['backward_type'] = 'sum_backward'
            else:
                item.grad = 0.
                item._node_attr['backward_type'] = None

        #设置节点状态
        self.is_backward = True
        self.is_forward = False
        pass

if __name__ == '__main__':

    input1 = data_node().initialize(3)
    input2 = data_node().initialize(4)
    input4 = data_node().initialize(5)

    inputs = [input1,input2,input4]

    input3 = data_node()
    input3.grad = 0.4

    adder = sum_node(inputs,input3)
    adder.forward()

    print(input3.data)

    adder.backward()

    print(inputs[0].grad)
    print(inputs[1].grad)
    print(inputs[2].grad)



