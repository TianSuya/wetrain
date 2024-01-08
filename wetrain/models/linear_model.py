import os.path
import sys
import numpy as np


from wetrain.models.model_bases.base_model import base_model
from wetrain.wtensor import wtensor
from wetrain.operator import matmul

class Linear(base_model):

    def __init__(self,in_dim,out_dim):

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = wtensor(np.random.normal(size=(in_dim,out_dim)))

    def params(self):
        return [self.weight]

    def forward(self,x:wtensor):

        if np.shape(x.data)[-1] != self.in_dim:
            raise ValueError('Input dim not equal to in_dim')

        self.weight.clear_rela() #由于weight是有会被保留的wtensor，所以每次前向传播都需要清除节点信息

        ans = matmul(x,self.weight)
        return ans


if __name__ == '__main__':
    model = Linear(3,2)
    c = wtensor([[1,2,3],[3,4,5]])
    b = model(c)
    print(b.data)
    b.backward()