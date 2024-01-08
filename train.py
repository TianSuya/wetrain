import sys

from wetrain.models.linear_model import Linear
from wetrain.models.mse_loss import MSELoss
from wetrain.optimizer.sgd_opt import SGD
from boston_data import my_data
from wetrain.wtensor.wtensor import wtensor

my_dataset = my_data()

epoch = 500
if __name__ == '__main__':

    model = Linear(4,1)
    optimizer = SGD(model.params(),learning_rate=0.01)
    loss = MSELoss()

    for epoch in range(epoch):
        epoch_loss = []
        for data,label in my_dataset:

            input = wtensor(data)
            labels = wtensor(label)


            output = model(input)

            # print(output.data)

            the_loss = loss(output,labels)
            epoch_loss.append(the_loss.data)

            optimizer.zero_grad()
            the_loss.backward()
            optimizer.step()

        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        print('in this batch,the loss is:',epoch_loss)









