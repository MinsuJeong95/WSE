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

import pickle
import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset
from collections import defaultdict


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

def test(datasetType, Fold, modelTypes, modelType, models, gceNet, DBPath, numOfClass, iterCheck=0, splitChannel=0):

    batchSize = 256

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    imgPre = imgPreprocess.imgPreprocess()

    model1 = models[0].convnext_small(pretrained=True, num_classes=numOfClass)
    model1Type = modelTypes[0]
    model2 = models[1].convnext_small(pretrained=True, num_classes=numOfClass)
    model2Type = modelTypes[1]


    bestModel1Path = './' + datasetType + '\\bestModel' + '\\' + model1Type + '\\' + Fold
    bestModel1Name = os.listdir(bestModel1Path)[0]
    # model1TrainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\' + 'allcam'

    bestModel2Path = './' + datasetType + '\\bestModel' + '\\' + model2Type + '\\' + Fold
    bestModel2Name = os.listdir(bestModel2Path)[0]
    # model2TrainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\' + 'allcam'

    # Fold = Fold + '_' + str(iterCheck)
    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    allCamGalleryTestset = CustomDataset.CustomDataset(root_dir=DBPath,
                                                      transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryTestset, batch_size=batchSize, shuffle=False, pin_memory=True)

    # Class 개수 계산
    allCamProbClassLen = len(allCamGalleryTestset)  # No geometric
    totalLen = allCamProbClassLen

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

    h1 = model1.stages[-1].register_forward_hook(save_gradient1)
    h2 = model2.stages[-1].register_forward_hook(save_gradient2)
    h_att = model2.stages[-1][3].SpatialGate.spaSig.register_forward_hook(save_gradient_att)

    print('Fold : ' + Fold)
    path = '.\\' + datasetType + '\\valResult\\' + modelType + '\\' + Fold + '\\' + 'selectEpoch'
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
                galleryLabels = []
                gallerySaveNames = []
                galleryImgs = galleryData['image'].to(device)
                galleryNames = galleryData['filename']

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

                outputs, feature = model(totalData)

                progress = progress + outputs.shape[0]
                print("load : %.5f%%" % ((progress * 100) / totalLen))

                for names in galleryNames:
                    nameParsing = names.split('\\')
                    galleryLabels.append(nameParsing[len(nameParsing)-2])
                    gallerySaveNames.append(nameParsing[len(nameParsing)-1])

                for result_i in range(len(feature)):
                    ReIDresult.append([galleryLabels[result_i], gallerySaveNames[result_i], feature[result_i]])

                ReIDdict = ReIDresult

        reidResult = defaultdict(list)

        if datasetType == 'SYSU-MM01_thermal':
            #center image 정하기
            centerImg = []
            imgTmp = []
            calTmp = []

            keyTmp = 0
            for i, (key_cen, name_cen, value_cen) in enumerate(ReIDdict):
                key_cen = int(key_cen)
                if keyTmp != 0 and keyTmp != key_cen:
                    for cen_i, (key_tmp, name_tmp, value_tmp) in enumerate(imgTmp):
                        l2AllVal = 0
                        for cen2_i, (key_tmp2, name_tmp2, value_tmp2) in enumerate(imgTmp):
                            if name_tmp == name_tmp2:
                                continue
                            l2Value = torch.dist(value_tmp, value_tmp2, 2)

                            l2AllVal = l2AllVal + l2Value
                        calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                    calTmp.sort(key=lambda x: -x[3])
                    centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])
                    # centerImg.append([calTmp[4][0], calTmp[4][1], calTmp[4][2]])
                    calTmp = []
                    imgTmp = []
                keyTmp = key_cen
                imgTmp.append([key_cen, name_cen, value_cen])

            if imgTmp != []:
                for cen_i, (key_tmp, name_tmp, value_tmp) in enumerate(imgTmp):
                    l2AllVal = 0
                    for cen2_i, (key_tmp2, name_tmp2, value_tmp2) in enumerate(imgTmp):
                        if name_tmp == name_tmp2:
                            continue
                        l2Value = torch.dist(value_tmp, value_tmp2, 2)

                        l2AllVal = l2AllVal + l2Value
                    calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                calTmp.sort(key=lambda x: -x[3])
                centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])
                # centerImg.append([calTmp[4][0], calTmp[4][1], calTmp[4][2]])
                calTmp = []
                imgTmp = []

            for i, (key_gal, name_gal, value_gal) in enumerate(centerImg):
                key_gal = int(key_gal)
                print('gal_cnt : ', i)
                for j, (key_prob, name_prob, value_prob) in enumerate(ReIDdict):
                    key_prob = int(key_prob)
                    if key_gal == key_prob and name_gal == name_prob:
                        continue

                    value_cal = torch.dist(value_gal, value_prob, 2)

                    reidResult[key_gal].append([key_prob, name_prob, value_cal])

        else:
            for i, (key_gal, name_gal, value_gal) in enumerate(ReIDdict):
                key_gal = int(key_gal)
                print('gal_cnt : ', i)
                for j, (key_prob, name_prob, value_prob) in enumerate(ReIDdict):
                    key_prob = int(key_prob)
                    if key_gal == key_prob and name_gal == name_prob:
                        continue

                    value_cal = torch.dist(value_gal, value_prob, 2)

                    reidResult[key_gal].append([key_prob, name_prob, value_cal])


        for key_gal in reidResult.keys():
            reidResult[key_gal].sort(key=lambda x: x[2])

        ReIDdict = reidResult

        saveTestResultName = modelPath.split('.')[0]
        testPath = './' + datasetType + '\\' + 'testResult/' + modelType + '\\' + Fold + '\\' + '/epochTermTest/'
        if not os.path.isdir(testPath):
            os.makedirs(testPath)
        with open(testPath + 'testResult_' + saveTestResultName + '.pickle',
                  'wb') as fw:
            pickle.dump(ReIDdict, fw)

    del allCamGalleryTestset
    del allCamGalleryLoader
    torch.cuda.empty_cache()
