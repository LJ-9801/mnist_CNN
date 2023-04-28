import os
from dataloader import MnistDataloader
import numpy as np
    


path  = os.getcwd()+'/archive'
#training images and labels
training_images_filepath = os.path.join(path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = os.path.join(path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')

#testing images and labels
testing_images_filepath = os.path.join(path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
testing_labels_filepath = os.path.join(path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                      testing_images_filepath, testing_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

print("finish loading data!")

from model import mnistNet
import torch
import torch.nn as nn

model = mnistNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()


x_train = torch.stack([torch.Tensor(np.ndarray([i])) for i in x_train])
y_train = torch.stack([torch.Tensor(np.ndarray([i])) for i in y_train])

x_test = torch.stack([torch.Tensor([i]) for i in x_test])
y_test = torch.stack([torch.Tensor([i]) for i in y_test])

from tqdm import tqdm as tq

print("start training!")

batch_size = 32

losses = []

for i in tq(range(1000)):
    sample = torch.randint(0, len(x_train), size=(batch_size,))
    x_train_batch = x_train[sample]
    y_train_batch = y_train[sample].reshape(batch_size).long()
    optimizer.zero_grad()
    output = model(x_train_batch)
    loss_val = loss(output, y_train_batch)
    losses.append(loss_val.item())
    loss_val.backward()
    optimizer.step()
    #if i % 100 == 0:
    #    print('Epoch: {}, Loss: {:.4f}'.format(i, loss_val.item()))

print("finish training!")

error = 0
for i in tq(range(len(x_test))):
    predict = model(x_test[i].reshape(1, 1, 28, 28))
    truth = int(y_test[i])
    predict = int(torch.argmax(predict))
    error += (predict != truth)

print("error rate: ", error/len(x_test))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()


