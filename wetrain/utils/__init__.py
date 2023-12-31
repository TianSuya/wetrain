import os,sys
path = os.path.dirname(__file__)
sys.path.append(path)

from backward import BackWard
from forward import Forward
from zero_grad import ZeroGrad
from dataset import Dataset