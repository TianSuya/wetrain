import sys

sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../layers')

from layers.node_layer import data_layer

class ZeroGrad(object):

    def __init__(self):
        pass

    def __call__(self,end:data_layer):
        zero_queue = []
        for item in end:
            zero_queue.append(item)

        while len(zero_queue) > 0:
            item = zero_queue.pop(0)
            item.zero_grad()
            zero_queue.extend(item.children)
