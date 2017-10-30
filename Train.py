import FCN as fcn
import DataDownloader_wuweijia as data_loader

from skimage import img_as_ubyte
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.optim as optim

root_dir_wuweijia = './20171020_morning/'

transformed_dataset = data_loader.number_dataset(root_dir=root_dir_wuweijia, \
                                                       transform=data_loader.transforms.Compose([data_loader.ToTensor(),\
                                                                                                 # data_loader.Slice(0), \
                                                        data_loader.Normalize([0, 0, 0], [1, 1, 1])]))


train_data_loader = data_loader.DataLoader(transformed_dataset, batch_size=2,
                        shuffle=True, num_workers=3)


net = fcn.Fcn()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
my_loss_data = []
standard_loss_data = []

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def show_output_data(outputs, mean, dev):
    outputs_data = outputs.view([-1, 54, 54])
    # demo = predicted.data.numpy();
    outputs_data = outputs_data.data.numpy()[0, :, :]
    outputs_data[outputs_data > (mean + dev)] = 0
    outputs_data[outputs_data < (mean - dev)] = 0
    return outputs_data

def show_picture(img):
    img= img * 50
    viewer = ImageViewer(img_as_ubyte(img))
    viewer.show()

for epoch in range(5):  # loop over the dataset multiple times

    loss_recorder = []
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs
        # data = train_data_loader[i]
        inputs, labels = data['image'], data['groundtruth_image']
        # inputs = image
        # labels = groundtruth_image

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels.type(data_loader.torch.LongTensor))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = cross_entropy2d(outputs, labels)
        # loss /= len()
        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        loss_recorder.append(loss.data[0])

        if i % 240 == 239:    # print every 120 mini-batches
            print('[%d, %5d] loss: %.30f' %
                  (epoch + 1, i + 1, loss.data[0]))
            running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1)
            pre = predicted.numpy()
            show_picture(pre[0])
            show_picture(labels.data.numpy()[0])
            # show_picture(pre[1])
            # show_picture(pre[2])
            # show_picture(pre[3])

            output_debug = labels.data.numpy()
            predicted = show_output_data(outputs, 3, 0.3)
            labels = show_output_data(labels, 3, 0.3)
            plt.plot(range(len(loss_recorder)), loss_recorder)




            # a = (a - a.min()) / (a.max() - a.min())
            # viewer = ImageViewer(img_as_ubyte(a))
            # viewer.show()

            # b = labels.view([-1, 54, 54])
            # b = b.data.numpy()[0, :, :]
            # b = (b - b.min())/(b.max() - b.min())
            # 
            # c = a - b
            # c = c**2
            # c = c.sum()
            # view_b = ImageViewer(img_as_ubyte(b))
            # view_b.show()
            # my_loss_data.append(c/2916.0)
            # standard_loss_data.append(loss.data[0])
    # plt.figure()
    # plt.plot(range(len(my_loss_data)), my_loss_data)
    # plt.xlabel("mini-batch times")
    # plt.ylabel("MSE loss")
    # plt.title("My mse loss plot")
    # 
    # plt.figure()
    # plt.plot(range(len(standard_loss_data)), standard_loss_data)
    # plt.xlabel("mini-batch times")
    # plt.ylabel("MSE loss")
    # plt.title("Pytorch mse loss plot")

print(i)
print('Finished Training')