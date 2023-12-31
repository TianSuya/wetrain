import sys

sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../layers')

from layers.node_layer import data_layer

class BackWard:

    def __init__(self):
        pass

    def __call__(self,end:data_layer):
        backward_queue = []
        for item in end:
            item.grad = 1. #最后一层对自己的导数永远是1.
            backward_queue.append(item)

        while len(backward_queue) > 0:
            item = backward_queue.pop(0)
            item.backward()
            backward_queue.extend(item.children)