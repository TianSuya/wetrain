import sys

sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../layers')

from layers.node_layer import data_layer

class Forward:

    def __init__(self):
        pass

    def __call__(self,begin:data_layer):
        forward_queue = []
        for item in begin:
            forward_queue.append(item)

        while len(forward_queue) > 0:
            item = forward_queue.pop(0)
            item.forward()
            forward_queue.extend(item.parent)