import sys
import os

path = os.path.dirname(__file__)
sys.path.append(path)

from layers import DataLayer
from nodes import sum_node
from nodes import mul_node

from models.bases.base_model import base_model

class LinearModel(base_model):

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

    def in_concat(self,input:DataLayer):

        if len(input) != self.input_dim:
            raise ValueError('input dimension mismatch')

        self.input_layer = input
        self.weights = []

        for index in range(self.output_dim):
            weight_layer = DataLayer(self.input_dim, require_initialize=True, require_train=True)
            self.weights.append(weight_layer)

        self.output_layer = DataLayer(self.output_dim, require_train=True)

        self.middle_res = []
        for index in range(self.output_dim):
            middle_layer = DataLayer(self.input_dim, require_initialize=False, require_train=True)
            self.middle_res.append(middle_layer)

        # 将各个层拼接在一起
        for index in range(self.output_dim):
            sum_list = []

            for key in range(self.input_dim):
                mul_node(self.weights[index][key], self.input_layer[key], self.middle_res[index][key])
                sum_list.append(self.middle_res[index][key])
            sum_node(sum_list, self.output_layer[index])

    def __getitem__(self, idx):
        return self.output_layer[idx]

    def out_concat(self):
        return self.output_layer

    def params(self):
        return self.weights






