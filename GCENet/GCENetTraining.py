import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as nnf
import cv2
import os
import numpy as np
import random

import pickle
import utils.imgPreprocess as imgPreprocess
import shutil
import utils.CustomDataset as CustomDataset


def accuracy(out, yb):
    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    yb = yb.cpu()

    compare = []
    for i in range(len(out)):
        outList = out[i].tolist()
        tmp = max(outList)
        index = outList.index(tmp)
        compare.append(index)
    compare = torch.Tensor(compare).long()

    return (compare == yb).float().mean()


def custom_loader(path):
    ret = np.load(path)
    return ret


# 23-06-23 JMS 수정
def choiceData(labels, feature, device):
    totalAnchor = []
    totalPositive = []
    totalNegative = []

    labels = np.array(labels.to('cpu'))

    for labels_i in range(len(labels)):
        if len(set(labels)) != len(labels):
            sameLabelIdx = np.where(labels[labels_i] == labels)[0]
            differentLabelIdx = [i for i in range(len(labels)) if i not in sameLabelIdx]

            anchor = feature[labels_i]
            if len(sameLabelIdx) != 1:
                sameLabelIdx = np.delete(sameLabelIdx, np.where(sameLabelIdx == labels_i))
                positive = feature[random.choice(sameLabelIdx)]
            else:
                positive = feature[sameLabelIdx[0]]
            negative = feature[random.choice(differentLabelIdx)]
        else:
            anchor = feature[labels_i]
            positive = anchor
            otherIdx = list(range(len(labels)))
            otherIdx.pop(labels_i)
            negative = feature[random.choice(otherIdx)]

        totalAnchor.append(anchor)
        totalPositive.append(positive)
        totalNegative.append(negative)

    return torch.stack(totalAnchor, 0), torch.stack(totalPositive, 0), torch.stack(totalNegative, 0)
#####


def distanceAverage(gradient, channel, batch):
    spGrad = gradient.split(32, dim=1)
    spGrad = torch.stack(spGrad, dim=1)
    spGrad = spGrad.split(int(768/channel), dim=2)
    spGrad = torch.stack(spGrad, dim=1)
    gradResult = spGrad.mean(dim=3)

    gradResult = torch.reshape(gradResult, (batch, -1, 7, 7))
    return gradResult


def training(datasetType, Fold, modelType, gceNet, DBPath, numOfClass, numEpoch=1, startEpoch=0, lr=1e-5, wd=1e-4, splitChannel=0):
    # Fold = Fold + '_' + str(iterCheck)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    batchSize = 12
    learningRate = lr
    wdecay = wd
    model = gceNet.GCENet(inputChannel=splitChannel*2+1+1024)
    model.fc = nn.Linear(splitChannel*2+1+1024, numOfClass)

    imgPre = imgPreprocess.imgPreprocess()

    print('Fold : ' + Fold)

    pathLen = len(DBPath.split('\\'))

    trans = transforms.Compose([transforms.ToTensor()])
    Trainset = torchvision.datasets.DatasetFolder(root=DBPath, loader=custom_loader, extensions='.npy', transform=trans)
    Loader = DataLoader(Trainset, batch_size=batchSize, shuffle=True, pin_memory=True)

    model.train()  # 학습모드
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterionTriplet = nn.TripletMarginLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=wdecay)

    iterCnt = 0
    fileName = 0
    lastEpoch = 0
    saveIterCnt = []
    saveLoss = []
    saveAccuracy = []

    for epoch in range(numEpoch - startEpoch):  # 데이터셋을 수차례 반복합니다.
        for i, Data in enumerate(Loader):
            inputImgs = Data[0].to(device)
            labels = Data[1].to(device)

            if splitChannel != 768:
                #input split 후 average
                spFeature = inputImgs.split(768 * 2 + 1, dim=1)
                patchImg = spFeature[1]
                spFeature = spFeature[0].split(768, dim=1)
                oriFreature = spFeature[0]
                attFeature = spFeature[1]
                cbamFeature = spFeature[2]

                oriFreature = distanceAverage(oriFreature, splitChannel, inputImgs.size(dim=0))
                attFeature = distanceAverage(attFeature, splitChannel, inputImgs.size(dim=0))
                inputImgs = torch.cat([oriFreature, attFeature, cbamFeature, patchImg],
                                      dim=1)

            optimizer.zero_grad()
            outputs, tpFeature = model(inputImgs)
            anchor, positive, negative = choiceData(labels, tpFeature, device)
            ceLoss = criterion(outputs, labels)
            tpLoss = criterionTriplet(anchor, positive, negative)
            loss = ceLoss + tpLoss
            loss.backward()
            optimizer.step()

            # 통계 출력
            acc = accuracy(outputs, labels)
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, loss.item()))

            saveIterCnt.append(iterCnt)
            saveLoss.append(loss.item())
            saveAccuracy.append(acc)

            iterCnt = iterCnt + 1

        fileName = DBPath.split('\\')[pathLen-4] + '_' + DBPath.split('\\')[pathLen-3] + '_' + \
                   DBPath.split('\\')[pathLen-1]
        PATH = './' + datasetType + '/trainModels/' + modelType + '/' + \
               Fold + '/epochTermModel/'
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        trainPath = PATH + fileName + '_ReID_' + modelType + '_' + \
               str(epoch + startEpoch + 1) + '.pth'

        print(trainPath)
        torch.save(model.state_dict(), trainPath)

        lastEpoch = epoch + 1

        trainInfoPath = './' + datasetType + '/trainModels/' + modelType + '/' + Fold + \
                        '/saveEpochInfo/'
        if not os.path.isdir(trainInfoPath):
            os.makedirs(trainInfoPath)

        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveIterCnt_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveIterCnt, fw)
        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveLoss_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveLoss, fw)
        with open(trainInfoPath + fileName + '_ReID_' + modelType + '_saveAccuracy_' + str(
                startEpoch + 1) + '-' + str(startEpoch + lastEpoch) + '.pickle', 'wb') as fw:
            pickle.dump(saveAccuracy, fw)

    del Trainset
    del Loader
    torch.cuda.empty_cache()

    print('Finished Training')


