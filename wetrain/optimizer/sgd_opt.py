import os,sys

import numpy as np

path = os.path.dirname(__file__)
sys.path.append(path)

class SGD(object):

    def __init__(self,params:list,learning_rate:float):

        self.params = params
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            param.data -= param.grad * self.learning_rate

    def zero_grad(self):

        for param in self.params:
            param.grad = np.zeros_like(param.data)