import torch.nn as nn
import torch.nn.functional as F
import torch
import models.ECA as ECA

class groupConv(nn.Module):
    def __init__(self, inputChannel):
        super().__init__()
        self.conv1 = nn.Conv2d(inputChannel, int(inputChannel / 2), kernel_size=1, stride=1)
        self.batch1 = nn.BatchNorm2d(int(inputChannel / 2))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(inputChannel / 2), int(inputChannel / 2), kernel_size=3, stride=1, groups=2 ** 5,
                               padding=1)
        self.batch2 = nn.BatchNorm2d(int(inputChannel / 2))
        self.conv3 = nn.Conv2d(int(inputChannel / 2), inputChannel, kernel_size=1, stride=1)
        self.batch3 = nn.BatchNorm2d(inputChannel)
        self.eca = ECA.eca_layer(channel=inputChannel)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.eca(x)
        x += shortcut
        x = self.relu(x)

        return x

class GCENet(nn.Module):
    def __init__(self, inputChannel=4417):
        super().__init__()
        self.layer1 = groupConv(inputChannel)
        self.layer2 = groupConv(inputChannel)
        self.layer3 = groupConv(inputChannel)
        # self.layer4 = groupConv(inputChannel)
        # self.layer5 = groupConv(inputChannel)
        self.fc = nn.Linear(inputChannel, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        feature = x
        x = self.fc(x)

        return x, feature
