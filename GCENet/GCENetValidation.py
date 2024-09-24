import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as nnf
import cv2
import os
import torch.nn.functional as F
import numpy as np
import random

import pickle
import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset

gradient1 = []
gradient2 = []
gradient_att = []


def save_gradient1(*args):
    # print("original : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient1.append(grad_output)  # forward
    # print(self.gradient[0].size())


def save_gradient2(*args):
    # print("attention : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient2.append(grad_output)  # forward
    # print(self.gradient[0].size())


def save_gradient_att(*args):
    # print("CBAM : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient_att.append(grad_output)  # forward
    # print(self.gradient[0].size())


gradient = []


def save_gradient(*args):
    # print("distanceCal : ", "Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient.append(grad_output)  # forward
    # print(self.gradient[0].size())


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
            if len(differentLabelIdx) == 0:
                negative = feature[random.choice(sameLabelIdx)]
            else:
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

def imgSplit(img, batch):
    img = torchvision.transforms.Grayscale()(img)
    splitImgs = img.split(7, dim=2)
    splitImgs = torch.stack(splitImgs, dim=2)
    splitImgs = splitImgs.split(7, dim=4)
    splitImgs = torch.stack(splitImgs, dim=2)

    splitImgs = torch.reshape(splitImgs, (batch, -1, 7, 7))

    return splitImgs

def validation(datasetType, Fold, modelTypes, modelType, models, gceNet, DBPath, numOfClass, iterCheck=0, splitChannel=0):
    batchSize = 256
    # batchSize = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    model1Type = modelTypes[0]
    model2Type = modelTypes[1]

    bestModel1Path = './' + datasetType + '\\bestModel' + '\\' + model1Type + '\\' + Fold
    bestModel1Name = os.listdir(bestModel1Path)[0]
    model1ValDBPath = DBPath

    bestModel2Path = './' + datasetType + '\\bestModel' + '\\' + model2Type + '\\' + Fold
    bestModel2Name = os.listdir(bestModel2Path)[0]
    model2ValDBPath = DBPath

    # trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # allCamProbValset = CustomDataset.CustomDataset(root_dir=model1ValDBPath,
    #                                                transforms=trans)
    # allCamProbLoader = DataLoader(allCamProbValset, batch_size=batchSize, shuffle=True, pin_memory=True)
    allCamGalleryValset = CustomDataset.CustomDataset(root_dir=model2ValDBPath,
                                                      transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryValset, batch_size=batchSize, shuffle=True, pin_memory=True)

    # Class 개수 계산
    allCamProbClassLen = len(allCamGalleryValset)  # No geometric
    totalLen = allCamProbClassLen

    model1 = models[0].convnext_small(pretrained=True, num_classes=numOfClass)
    model2 = models[1].convnext_small(pretrained=True, num_classes=numOfClass)

    model1.load_state_dict(torch.load(
        bestModel1Path + '\\' + bestModel1Name),
        strict=False)
    model2.load_state_dict(torch.load(
        bestModel2Path + '\\' + bestModel2Name),
        strict=False)

    model1.eval()
    model1.to(device)
    model2.eval()
    model2.to(device)
    m = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss().to(device)
    criterionTriplet = nn.TripletMarginLoss(margin=10.0).to(device)

    h1 = model1.stages[-1].register_forward_hook(save_gradient1)
    h2 = model2.stages[-1].register_forward_hook(save_gradient2)
    h_att = model2.stages[-1][3].SpatialGate.spaSig.register_forward_hook(save_gradient_att)

    print('Fold : ' + Fold)
    path = '.\\' + datasetType + '\\trainModels\\' + modelType + '\\' + Fold + '\\' + 'epochTermModel'
    modelPaths = os.listdir(path)

    # 모델list
    modelNum = []
    for i in range(len(modelPaths)):  # 모델list 재배열
        modelName = modelPaths[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = modelPaths[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pth')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        modelPaths[num_i] = saveName

    print(len(modelPaths))

    for modelPath in modelPaths:
        ReIDdict = {}
        progress = 0
        model = gceNet.GCENet(inputChannel=splitChannel*2+1+1024)
        model.fc = nn.Linear(splitChannel*2+1+1024, numOfClass)
        model.load_state_dict(torch.load(path + '\\' + modelPath))
        model.eval()  # 평가모드
        model.to(device)

        with torch.no_grad():
            ReIDresult = []
            keyTmp = 0

            # allCamProb : 인식 / allCamGallery : 등록
            for galleryI, galleryData in enumerate(allCamGalleryLoader):
                finalLabels = []
                galleryLabels = galleryData['label'].to(device)
                gallerySaveNames = []
                gallerySaveLabels = []
                galleryImgs = galleryData['image'].to(device)
                galleryNames = galleryData['filename']

                for i in range(len(galleryNames)):
                    galleryRealName = galleryNames[i].split('\\')
                    galleryRealNameLen = len(galleryRealName)
                    galleryRealLabel = galleryRealName[galleryRealNameLen - 2]
                    gallerySaveName = galleryRealName[galleryRealNameLen - 3] + '_' + \
                                      galleryRealName[galleryRealNameLen - 2] + '_' + \
                                      galleryRealName[galleryRealNameLen - 1].split('.')[0]

                    gallerySaveNames.append(gallerySaveName)
                    gallerySaveLabels.append(galleryRealLabel)

                # if galleryImgs.shape[1] == 1:  # IPVT2일 경우 Channel 맞춰줌
                #     reShape = torch.cat([galleryImgs, galleryImgs, galleryImgs], dim=1)
                #     galleryImgs = reShape

                outputs1, _ = model1(galleryImgs)
                outputs1 = m(outputs1)

                outputs2, _ = model2(galleryImgs)
                outputs2 = m(outputs2)

                if splitChannel != 768:
                    splitGrad1 = distanceAverage(gradient1[0], splitChannel, galleryImgs.size(dim=0))
                    splitGrad2 = distanceAverage(gradient2[0], splitChannel, galleryImgs.size(dim=0))
                else:
                    splitGrad1 = gradient1[0]
                    splitGrad2 = gradient2[0]

                patchImg = imgSplit(galleryImgs, galleryImgs.size(dim=0))

                totalData = torch.cat([splitGrad1, splitGrad2, gradient_att[0], patchImg],
                                      dim=1)

                gradient1.clear()
                gradient2.clear()
                gradient_att.clear()

                # 23-10-30 loss 수정확인
                outputs, tpFeature = model(totalData)
                anchor, positive, negative = choiceData(galleryLabels, tpFeature, device)
                tpLoss = criterionTriplet(anchor, positive, negative)
                loss = tpLoss

                progress = progress + outputs.shape[0]
                print("load : %.5f%%" % ((progress * 100) / totalLen))
                print('loss : %.6f' %
                      (loss.item()))

                for labelCnt in range(outputs.shape[0]):
                    label = int(gallerySaveLabels[labelCnt])

                    ReIDkey = label

                    if keyTmp != ReIDkey:
                        value = ReIDdict.get(keyTmp)
                        if value != None:
                            for valueCnt in range(len(ReIDresult)):
                                value.append(ReIDresult[valueCnt])
                            ReIDdict[keyTmp] = value
                        else:
                            ReIDdict[keyTmp] = ReIDresult

                        ReIDresult = []
                    ReIDresult.append([tpFeature[labelCnt], loss.item(), ReIDkey])
                    keyTmp = ReIDkey

            value = ReIDdict.get(keyTmp)
            if value != None:
                for valueCnt in range(len(ReIDresult)):
                    value.append(ReIDresult[valueCnt])
                ReIDdict[keyTmp] = value
            else:
                ReIDdict[keyTmp] = ReIDresult

            saveValResultName = modelPath.split('.')[0]
            valResultPath = './' + datasetType + '\\' + 'valResult/' + modelType + '\\' + Fold + '/epochTermValidation/'
            if not os.path.isdir(valResultPath):
                os.makedirs(valResultPath)
            with open(valResultPath + 'valResult' + '_' + saveValResultName + '.pickle',
                      'wb') as fw:
                pickle.dump(ReIDdict, fw)

    del allCamGalleryValset
    del allCamGalleryLoader
    torch.cuda.empty_cache()

