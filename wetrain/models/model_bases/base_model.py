
class base_model:

    def __init__(self):
        pass

    def forward(self,*param):
        pass

    def __call__(self,*param):
        ans = self.forward(*param)
        for item in param:
            item.forward()
        return ans

if __name__ == '__main__':

    a = base_model()
    a('hello','yes')