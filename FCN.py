import torch.nn as nn


class Fcn(nn.Module):
    def __init__(self):
        super(Fcn, self).__init__()

        #conv1
        self.conv_1 = nn.Conv2d(3, 16, 5)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Conv2d(16, 64, 4)
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Conv2d(64, 512, 4)
        self.relu_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(2, 2)

        self.fc_1 = nn.Conv2d(512, 512, 4)
        self.relu_4 = nn.ReLU()
        # self.dropout = nn.Dropout2d()

        self.score_fr = nn.Conv2d(512, 5, 1)
        self.upscore = nn.ConvTranspose2d(5, 5, 54, stride=32, bias=False)

    def forward(self, x):
        result = x
        result = self.relu_1(self.conv_1(result))
        result = self.pool_1(result)

        result = self.relu_2(self.conv_2(result))
        result = self.pool_2(result)

        result = self.relu_3(self.conv_3(result))
        result = self.pool_3(result)

        result = self.relu_4(self.fc_1(result))


        result = self.score_fr(result)
        result = self.upscore(result)

        return result