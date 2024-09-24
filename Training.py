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

import pickle
import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset
import shutil
import random


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


def choiceData(labels, feature, device):
    idxSize = len(labels)
    anchor = 0
    positive = 0
    negative = 0

    labels = np.array(labels.to('cpu'))

    if len(set(labels)) != len(labels):
        visited = set()
        dup = [x for x in labels if x in visited or (visited.add(x) or False)]
        sameLabelIdx = np.where(labels == dup[0])[0]

        anchor = feature[sameLabelIdx[0]]
        positive = feature[sameLabelIdx[1]]

        for i in range(idxSize):
            if i != sameLabelIdx[0] and i != sameLabelIdx[1]:
                negative = feature[i]
                break

    else:
        anchorIdx = int(np.random.choice(idxSize, 1, replace=False))
        anchorLabel = labels[anchorIdx]
        anchor = feature[anchorIdx]

        for i in range(idxSize):
            if labels[i] != anchorLabel:
                negative = feature[i]
                break

        positive = anchor

    return anchor, positive, negative


def training(datasetType, modelType, Fold, insertModel, DBPath, numOfClass, numEpoch=1, startEpoch=0, lr=1e-3, wd=1e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    batchSize = 12
    learningRate = lr
    wdecay = wd
    print("numOfClass :", numOfClass)
    print('Fold : ' + Fold)

    model = insertModel.convnext_small(pretrained=True, num_classes=numOfClass)

    model.wsFlag = True  #operation weak saliency

    pathLen = len(DBPath.split('\\'))
    fileName = DBPath.split('\\')[pathLen - 4] + '_' + DBPath.split('\\')[pathLen - 3] + '_' + \
               DBPath.split('\\')[pathLen - 1]
    PATH = './' + datasetType + '/trainModels/' + modelType + '/' + \
           Fold + '/epochTermModel/'
    if not os.path.isdir(PATH):
        os.makedirs(PATH)
    loadTrainPath = PATH + fileName + '_ReID_' + modelType + '_' + \
                str(startEpoch) + '.pth'

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    Trainset = CustomDataset.CustomDataset(root_dir=DBPath, transforms=trans)
    Loader = DataLoader(Trainset, batch_size=batchSize, shuffle=True, pin_memory=True)

    if startEpoch != 0:
        model.load_state_dict(torch.load(loadTrainPath))

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterionTriplet = nn.TripletMarginLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=wdecay)

    iterCnt = 0
    saveIterCnt = []
    saveLoss = []
    saveAccuracy = []

    for epoch in range(numEpoch - startEpoch):
        for i, Data in enumerate(Loader):
            inputImgs = Data['image'].to(device)
            labels = Data['label'].to(device)
            optimizer.zero_grad()

            if model.wsFlag == True:
                outputs, WSoutputs, tpFeature = model(inputImgs)    # ws on
                ceLoss = criterion(outputs, labels) + criterion(WSoutputs, labels)  # ws on
            else:
                outputs, tpFeature = model(inputImgs)   # ws off
                ceLoss = criterion(outputs, labels)  #ws off

            anchor, positive, negative = choiceData(labels, tpFeature, device)
            tpLoss = criterionTriplet(anchor, positive, negative)
            loss = ceLoss + tpLoss
            loss.backward()
            optimizer.step()

            acc = accuracy(outputs, labels)
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, loss.item()))
            saveIterCnt.append(iterCnt)
            saveLoss.append(loss.item())
            saveAccuracy.append(acc)

            iterCnt = iterCnt + 1

        trainPath = PATH + fileName + '_ReID_' + modelType + '_' + \
               str(epoch + startEpoch + 1) + '.pth'
        print(trainPath)
        torch.save(model.state_dict(), trainPath)
        lastEpoch = epoch + 1

        trainInfoPath = './' + datasetType + '/trainModels/' + modelType + '/' + Fold \
                        + '/saveEpochInfo/'

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
