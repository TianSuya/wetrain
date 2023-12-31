import os
import random
import sys
path = os.path.dirname(__file__)
sys.path.append(path+'/..')
from nodes import data_node

class node_layer:

    def __init__(self,node_num):

        self.node_num = node_num
        self.nodes = []

    def __getitem__(self,idx):
        return self.nodes[idx]

    def __len__(self):
        return self.node_num

class data_layer(node_layer):

    def __init__(self,node_num,require_initialize = False,require_train = True):

        super().__init__(node_num)

        if require_initialize:
            initialize = [random.gauss(0.,1.) for i in range(node_num)]
            for i in range(node_num):
                self.nodes.append(data_node(initialize[i],require_grad=require_train))
        else:
            for i in range(node_num):
                self.nodes.append(data_node(require_grad=require_train))

    def set_values(self,inputs):
        for index in range(self.node_num):
            self.nodes[index].set_value(inputs[index])

if __name__ == '__main__':

    node_layer = node_layer(10)

    for item in node_layer:
        print(item.data)