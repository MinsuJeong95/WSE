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

import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset

gradient1 = []
gradient2 = []
gradient_att = []
def save_gradient1(*args):
    # print("Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient1.append(grad_output)  # forward
    # print(self.gradient[0].size())

def save_gradient2(*args):
    # print("Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient2.append(grad_output)  # forward
    # print(self.gradient[0].size())

def save_gradient_att(*args):
    # print("Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient_att.append(grad_output)  # forward
    # print(self.gradient[0].size())

gradient = []
def save_gradient(*args):
    # print("Gradient saved!!!!")
    grad_input = args[1]
    grad_output = args[2]
    # print(grad_output)
    # self.gradient.append(grad_output[0]) #backward
    gradient.append(grad_output)  # forward
    # print(self.gradient[0].size())

def saveFeatureMap(datasetType, modelTypes, Fold, models, DBPath, numOfClass):
    batchSize = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    model1 = models[0].convnext_small(pretrained=True, num_classes=numOfClass)
    model1Type = modelTypes[0]
    model2 = models[1].convnext_small(pretrained=True, num_classes=numOfClass)
    model2Type = modelTypes[1]


    bestModel1Path = './' + datasetType + '\\bestModel' + '\\' + model1Type + '\\' + Fold
    bestModel1Name = os.listdir(bestModel1Path)[0]
    model1TrainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\' + 'allcam'

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    model1Trainset = CustomDataset.CustomDataset(root_dir=model1TrainDBPath,
                                                    transforms=trans)
    model1Loader = DataLoader(model1Trainset, batch_size=batchSize, shuffle=False, pin_memory=True)

    bestModel2Path = './' + datasetType + '\\bestModel' + '\\' + model2Type + '\\' + Fold
    bestModel2Name = os.listdir(bestModel2Path)[0]
    model2TrainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\' + 'allcam'

    # trans2 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Grayscale()])
    # model2Trainset = CustomDataset.CustomDataset(root_dir=model2TrainDBPath,
    #                                                 transforms=trans2)
    # model2Loader = DataLoader(model2Trainset, batch_size=batchSize, shuffle=False, pin_memory=True)

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
    # convnext
    h1 = model1.stages[-1].register_forward_hook(save_gradient1)
    h2 = model2.stages[-1].register_forward_hook(save_gradient2)
    h_att = model2.stages[-1][3].SpatialGate.spaSig.register_forward_hook(save_gradient_att)

    print('Fold : ' + Fold)

    # Class 개수 계산
    allCamProbClassLen = len(model1Trainset)
    # totalLen = allCamProbClassLen * len(allCamGalleryTestset)
    totalLen = allCamProbClassLen
    progress = 0
    with torch.no_grad():
        ReIDresult = []
        keyTmp = 0

        # allCamProb : 인식 / allCamGallery : 등록
        allCamProbiter1 = iter(model1Loader)
        # allCamProbiter2 = iter(model2Loader)
        for probI in range(allCamProbClassLen):
            probeData1 = allCamProbiter1.next()
            probImg1 = probeData1['image'].to(device)
            probName1 = probeData1['filename'][0].split('\\')
            probLabel1 = probeData1['label'].to(device)
            probNameLen1 = len(probName1)
            probRealLabel1 = probName1[probNameLen1 - 2]
            probSaveName1 = probName1[probNameLen1 - 3] + '_' + \
                           probName1[probNameLen1 - 2] + '_' + \
                           probName1[probNameLen1 - 1].split('.')[0]

            outputs1, _ = model1(probImg1)
            outputs1 = m(outputs1)

            outputs2, _ = model2(probImg1)
            outputs2 = m(outputs2)

            progress = progress + outputs1.shape[0]
            print("load : %.5f%%" % ((progress * 100) / totalLen))

            totalData = torch.cat([gradient1[0], gradient2[0], gradient_att[0]],
                                  dim=1)
            totalData = totalData.squeeze()
            totalData = totalData.transpose(0, 1)
            totalData = totalData.transpose(1, 2)

            gradient1.clear()
            gradient2.clear()
            gradient_att.clear()

            totalDataSavePath = datasetType + '\\GCENetTrainData\\' + Fold + '\\train\\' \
                                + probRealLabel1
            if not os.path.isdir(totalDataSavePath):
                os.makedirs(totalDataSavePath)
            np.save(totalDataSavePath + '\\' +
                    probName1[probNameLen1 - 1].split('.')[0],
                    totalData.cpu().numpy())




