# import sys
# lib_path = '/Users/weijiawu/PyTorchProjects/Img_Stitch'
# sys.path.insert(0, lib_path)
# print(sys.path)

# import Dataset_Generator as dg
import DataDownloader_wuweijia as data_loader
from skimage import img_as_ubyte
from skimage.viewer import ImageViewer
import numpy as np
import matplotlib.pyplot as plt

root_dir_wuweijia = './20171020_morning/'

transformed_dataset = data_loader.number_dataset(root_dir=root_dir_wuweijia, \
                                                       transform=data_loader.transforms.Compose([data_loader.ToTensor(),\
                                                                                                 # data_loader.Slice(0), \
                                                        data_loader.Normalize([0.485, 0.456,0.406], [0.229, 0.224,0.225])]))

train_data_loader = data_loader.DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=2)

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 30, 5)
        self.fc1 = nn.Linear(30 * 10 * 10, 54 * 54)
        self.fc2 = nn.Linear(54 * 54, 54 * 54)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
my_loss_data = []
standard_loss_data = []

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        # get the inputs
        inputs, labels = data['image'], data['groundtruth_image']
        # inputs = image
        # labels = groundtruth_image

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels.type(data_loader.torch.FloatTensor).div_(255))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

        # if i % 20 == 19:    # print every 20 mini-batches
        print('[%d, %5d] loss: %.30f' %
              (epoch + 1, i + 1, loss.data[0]))
        running_loss = 0.0
        predicted = outputs.view([-1, 54, 54])
        a = predicted.data.numpy()[0, :, :]
        a = (a - a.min()) / (a.max() - a.min())
        # viewer = ImageViewer(img_as_ubyte(a))
        # viewer.show()

        b = labels.view([-1, 54, 54])
        b = b.data.numpy()[0, :, :]
        b = (b - b.min())/(b.max() - b.min())

        c = a - b
        c = c**2
        c = c.sum()
        # view_b = ImageViewer(img_as_ubyte(b))
        # view_b.show()
        my_loss_data.append(c/2916.0)
        standard_loss_data.append(loss.data[0])
    plt.figure()
    plt.plot(range(len(my_loss_data)), my_loss_data)
    plt.xlabel("mini-batch times")
    plt.ylabel("MSE loss")
    plt.title("My mse loss plot")

    plt.figure()
    plt.plot(range(len(standard_loss_data)), standard_loss_data)
    plt.xlabel("mini-batch times")
    plt.ylabel("MSE loss")
    plt.title("Pytorch mse loss plot")


print('Finished Training')

# dataiter = iter(train_data_loader)
# images, labels = dataiter.next()
#
# outputs = net(Variable(images))
# predicted = outputs.numpy().reshape([-1, 54, 54])
# _, predicted = data_loader.torch.max(outputs.data, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(batch_size)))
#
# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))