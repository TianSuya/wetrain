import os
import sys

name = os.path.dirname(__file__)
sys.path.append(name)

from wetrain.wtensor.wtensor import wtensor
from nodes import add_node
from nodes import sub_node
from nodes import matmul_node
from nodes import cos_node
from nodes import div_node
from nodes import exp_node
from nodes import log_node
from nodes import pow_node
from nodes import sin_node
from nodes import sum_node
from nodes import tanh_node
from nodes import mul_node
def add(add_1:wtensor, add_2:wtensor):
    ans = wtensor()
    add_node(add_1,add_2,ans)
    return ans

def sub(sub_1:wtensor, sub_2:wtensor):
    ans = wtensor()
    sub_node(sub_1,sub_2,ans)
    return ans

def matmul(mul_1:wtensor, mul_2:wtensor):
    ans = wtensor()
    matmul_node(mul_1,mul_2,ans)
    return ans

def mul(mul_1:wtensor, mul_2:wtensor):
    ans = wtensor()
    mul_node(mul_1,mul_2,ans)
    return ans

def cos(input:wtensor):
    ans = wtensor()
    cos_node(input,ans)
    return ans

def div(div_1:wtensor, div_2:wtensor):
    ans = wtensor()
    div_node(div_1,div_2, ans)
    return ans

def exp(input:wtensor):
    ans = wtensor()
    exp_node(input,ans)
    return ans

def log(input:wtensor):
    ans = wtensor()
    log_node(input,ans)
    return ans

def pow(input:wtensor,n:float):
    ans = wtensor()
    pow_node(input,ans,n)
    return ans

def sin(input:wtensor):
    ans = wtensor()
    sin_node(input,ans)
    return ans

def sum(input:wtensor,dim=0):
    ans = wtensor()
    sum_node(input,ans,dim)
    return ans

def tanh(input:wtensor):
    ans = wtensor()
    tanh_node(input,ans)
    return ans








