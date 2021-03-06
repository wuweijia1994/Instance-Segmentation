import torch.nn as nn


class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()

        #conv1
        self.conv_1 = nn.Conv2d(3, 16, 5, padding=2)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Conv2d(16, 64, 5, padding=2)
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = nn.Conv2d(64, 128, 5, padding=2)
        self.relu_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool2d(2, 2)

        self.conv_4 = nn.Conv2d(128, 128, 5, padding=2)
        self.relu_4 = nn.ReLU()
        self.pool_4  = nn.MaxPool2d(2, 2)

        # self.conv_5 = nn.Conv2d(128, 128, 3)
        # self.relu_5 = nn.ReLU()

        # self.fc = nn.Conv2d(128, 128, 1)
        # self.relu_6 = nn.ReLU()

        # self.upscore_1 = nn.ConvTranspose2d(128, 128, 5, padding=1, stride=2)
        # self.relu_7 = nn.ReLU()

        self.upscore_2 = nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2)
        self.relu_8 = nn.ReLU()

        # self.score_fr = nn.Conv2d(128, 128, 1)
        self.upscore_3 = nn.ConvTranspose2d(128, 64, 7, padding=2, stride=2)
        self.relu_9 = nn.ReLU()
        self.upscore_4 = nn.ConvTranspose2d(64, 16, 5, padding=1, stride=2)
        self.relu_10 = nn.ReLU()
        self.upscore_5 = nn.ConvTranspose2d(16, 11, 4, padding=1, stride=2)


        # self.fc_1 = nn.Conv2d(512, 512, 4)
        # self.relu_4 = nn.ReLU()
        # self.dropout = nn.Dropout2d()


    def forward(self, x):
        result = x
        result = self.relu_1(self.conv_1(result))
        result = self.pool_1(result)
        result_conpool_1 = result

        result = self.relu_2(self.conv_2(result))
        result = self.pool_2(result)
        result_conpool_2 = result

        result = self.relu_3(self.conv_3(result))
        result = self.pool_3(result)

        result = self.relu_4(self.conv_4(result))
        result = self.pool_4(result)

        # result = self.relu_5(self.conv_5(result))

        # result = self.relu_6(self.fc(result))

        # result = self.relu_7(self.upscore_1(result))
        result = self.relu_8(self.upscore_2(result))

        # result = self.relu_4(self.fc_1(result))
        # result = self.score_fr(result)

        result = self.upscore_3(result)
        result = self.relu_9(result)

        result = result_conpool_2 + result
        result = self.upscore_4(result)
        result = self.relu_10(result)

        result = result_conpool_1 + result
        result = self.upscore_5(result)

        # result = result[:, :, 2:2 + x.data.shape[2], 2:2 + x.data.shape[3]]

        # result = x + self.conv_temp(result);
        # result = self.conv_final(result)

        return result