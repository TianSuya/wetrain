import os,sys
path = os.path.dirname(__file__)
sys.path.append(path)

class SgdOptimizer(object):

    def __init__(self,params:list,batch_size:int,learning_rate:float):

        self.params = params
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.count = 0

    def step(self):
        self.count += 1
        if self.count == self.batch_size:
            self.count = 0
            for param in self.params:
                for item in param:
                    # print('grad:',item.grad)
                    item.data -= self.learning_rate * item.grad