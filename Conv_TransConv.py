import torch.nn as nn


class C_TCn(nn.Module):
    def __init__(self):
        super(C_TCn, self).__init__()

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

        self.fc_1 = nn.Linear(128*6*6, 256)
        self.relu_4 = nn.ReLU()
        self.fc_2 = nn.Linear(256, 256)
        self.relu_5 = nn.ReLU()
        self.fc_3 = nn.Linear(256, 128*6*6)
        self.relu_6 = nn.ReLU()


        # self.score_fr = nn.Conv2d(128, 128, 1)
        self.upscore_1 = nn.ConvTranspose2d(128, 64, 7, padding=2, stride=2)
        self.relu_7 = nn.ReLU()
        self.upscore_2 = nn.ConvTranspose2d(64, 16, 5, padding=1, stride=2)
        self.relu_8 = nn.ReLU()
        self.upscore_3 = nn.ConvTranspose2d(16, 5, 4, padding=1, stride=2)


        # self.fc_1 = nn.Conv2d(512, 512, 4)
        # self.relu_4 = nn.ReLU()
        # self.dropout = nn.Dropout2d()
        #
        # self.score_fr = nn.Conv2d(512, 5, 1)
        # self.upscore = nn.ConvTranspose2d(5, 5, 54, stride = 32, bias=False)

        # self.upscore_2 = nn.ConvTranspose2d(5, 5, 10, stride=16, bias=False)
        # self.conv_temp = nn.Conv2d(5, 3, 1)
        # self.conv_final = nn.Conv2d(3, 5, 1)


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

        result = self.fc_1(result.view(-1, 128*6*6))
        result = self.relu_4(result)

        result = self.fc_2(result)
        result = self.relu_5(result)

        result = self.fc_3(result)
        result = self.relu_6(result)

        # result = self.relu_4(self.fc_1(result))
        # result = self.score_fr(result)

        result = self.upscore_1(result.view(-1, 128, 6, 6))
        result = self.relu_7(result)

        result = result_conpool_2 + result
        result = self.upscore_2(result)
        result = self.relu_8(result)

        result = result_conpool_1 + result
        result = self.upscore_3(result)

        # result = result[:, :, 2:2 + x.data.shape[2], 2:2 + x.data.shape[3]]

        # result = x + self.conv_temp(result);
        # result = self.conv_final(result)

        return result