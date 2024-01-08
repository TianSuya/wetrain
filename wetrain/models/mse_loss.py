import os.path
import sys
import numpy as np

name = os.path.dirname(__file__)
sys.path.append(name)

from model_bases.base_model import base_model
from wetrain.wtensor.wtensor import wtensor

from wetrain.operator.base import sub,pow,mul,sum

class MSELoss(base_model):

    def __init__(self):
        pass

    def forward(self,y_pred:wtensor,y_label:wtensor):
        the_sub = sub(y_pred,y_label)
        the_pow = pow(the_sub,2)
        the_mse = mul(the_pow,wtensor(np.zeros_like(y_pred.data)+0.5))
        the_sum = sum(the_mse,1)
        return the_sum

if __name__ == '__main__':

    loss = MSELoss()
    input1 = wtensor([[1,2,3],[4,5,6]])
    input2 = wtensor([[4,1,6],[2,4,8]])
    the_loss = loss(input1,input2)
    print('1',the_loss.data)
    the_loss.backward()
    print('2',input1.grad)



