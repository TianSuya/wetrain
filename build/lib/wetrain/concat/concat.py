import os.path
import sys

path = os.path.dirname(__file__)
sys.path.append(path)
sys.path.append(path+'/..')

class Concat:

    def __init__(self):
        pass

    def connect(self,models:list):
        out_layer = models[0]

        for item in models[1:]:
            item.in_concat(out_layer)
            out_layer = item.out_concat()
        pass

if __name__ == '__main__':

    model = Concat()


