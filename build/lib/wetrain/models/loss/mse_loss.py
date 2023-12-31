import sys
import os

path = os.path.dirname(__file__)
sys.path.append(path+'/../..')

from layers.node_layer import data_layer
from nodes.bases.data_node import data_node
from nodes.mul_node import mul_node
from wetrain.nodes.sub_node import sub_node
from nodes.pow_node import pow_node
from models.bases.base_model import base_model

class MSELoss(base_model):

    def __init__(self):
        pass

    def in_concat(self,nodes:data_layer):
        self.nodes = nodes
        self.label = data_layer(len(nodes),require_initialize=False,require_train=True)

        self.subs = []
        self.layer_1 = data_layer(len(nodes),require_initialize=False,require_train=True)
        for index in range(len(nodes)):
            self.subs.append(sub_node(self.label[index],self.nodes[index],self.layer_1[index]))

        self.layer_2 = data_layer(len(nodes),require_initialize=False,require_train=True)
        self.pows = []
        for index in range(len(nodes)):
            self.pows.append(pow_node(self.layer_1[index],self.layer_2[index],2))

        self.output = data_layer(len(nodes),require_initialize=False,require_train=True)
        self.muls = []
        for index in range(len(nodes)):
            self.muls.append(mul_node(data_node(0.5),self.layer_2[index],self.output[index]))

    def set_label(self,label:list):
        self.label.set_values(label)

    def get_loss(self):
        losses = []
        for item in self.output:
            losses.append(item.data)
        return sum(losses) / len(losses)

    def out_concat(self):
        return self.output

    def __getitem__(self, ids):
        return self.output[ids]

    def __len__(self):
        return len(self.output)

if __name__ == '__main__':

    model = MSELoss()



