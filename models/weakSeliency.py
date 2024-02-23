import torch
import torch.nn as nn
import time

class weakSaliency(nn.Module):
    def __init__(self, inputImgSize):
        super(weakSaliency, self).__init__()
        self.inputImgSize = inputImgSize

    def forward(self, x):
        batchFeatureA = x
        width = self.inputImgSize[0]
        hight = self.inputImgSize[1]
        resultFeature = 0
        batchResultFeature = 0

        # start = time.time()
        for i, featureA in enumerate(batchFeatureA):
            tmp = featureA.sum(dim=(1, 2))
            alpha = tmp/(hight * width)

            totalChannel = len(alpha)
            alpha = alpha.unsqueeze(1)
            alpha = alpha.unsqueeze(2)

            totalFeature = featureA * alpha
            totalFeature = totalFeature.sum(dim=0)

            resultFeature = totalFeature / torch.tensor(totalChannel)
            resultFeature = torch.unsqueeze(resultFeature, dim=0)

            if i == 0:
                batchResultFeature = resultFeature
            else:
                batchResultFeature = torch.cat([batchResultFeature, resultFeature], dim=0)

        batchResultFeature = torch.unsqueeze(batchResultFeature, dim=1)
        # end = time.time()
        # print(f"{end - start: .5f} sec")
        return batchResultFeature

# class scaleLayer(nn.Module):
#     def __init__(self, init_value=1e-3):
#         super(scaleLayer, self).__init__()
#         self.scale = nn.Parameter(torch.FloatTensor([init_value])).cuda()
#
#     def forward(self, x):
#         return x * self.scale


class weakSaliencyMechanism(nn.Module):
    # def __init__(self, outputChannel, inputImgSize, weakeningFactor=0.2, threshold=0.01):
    def __init__(self, outputChannel, inputImgSize, threshold=0.01):
        super(weakSaliencyMechanism, self).__init__()
        self.weakSaliency = weakSaliency(inputImgSize)
        # self.scale = 3
        # self.scaleLayer = scaleLayer()
        self.scaleLayer = nn.Parameter(torch.FloatTensor([1]))
        self.conv = nn.Conv2d(1, outputChannel, kernel_size=1, stride=1, bias=False).cuda()
        self.relu = nn.ReLU().cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        # self.wf = weakeningFactor
        self.wf = nn.Parameter(torch.FloatTensor([1]))
        self.th = threshold


    def forward(self, x):
        x_out = self.weakSaliency(x)
        # up_sample
        x_out = self.conv(x_out)
        featureMap = self.relu(x_out)

        # newFeature = 1 - (self.wf * self.sigmoid(self.scaleLayer(featureMap - self.th)))
        m = nn.Threshold(self.th, 0)
        featureMap = m(featureMap)
        newFeature = 1 - (self.wf * self.sigmoid(self.scaleLayer*featureMap))
        return newFeature



