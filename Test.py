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

def test(datasetType, modelType, Fold, insertModel, DBPath, numOfClass, iterCheck=0):
    batchSize = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    model = insertModel.convnext_small(pretrained=True, num_classes=numOfClass)  # convnext

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    allCamGalleryTestset = CustomDataset.CustomDataset(root_dir=DBPath,
                                              transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryTestset, batch_size=batchSize, shuffle=False, pin_memory=True)
    # Class 개수 계산
    totalLen = len(allCamGalleryTestset)

    print('Fold : ' + Fold)
    path = '.\\' + datasetType + '\\' + 'valResult\\' + modelType + '\\' + Fold + '\\' + 'selectEpoch'
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

        model.load_state_dict(torch.load(path + '\\' + modelPath))
        model.eval()  # 평가모드
        model.to(device)

        with torch.no_grad():
            ReIDresult = []
            for galleryI, galleryData in enumerate(allCamGalleryLoader):
                galleryLabels = []
                gallerySaveNames = []
                galleryImgs = galleryData['image'].to(device)
                galleryNames = galleryData['filename']

                _, features = model(galleryImgs)
                progress = progress + features.shape[0]
                print("load : %.5f%%" % ( (progress * 100) / totalLen ))

                for names in galleryNames:
                    nameParsing = names.split('\\')
                    galleryLabels.append(nameParsing[len(nameParsing)-2])
                    gallerySaveNames.append(nameParsing[len(nameParsing)-1])

                for result_i in range(len(features)):
                    ReIDresult.append([galleryLabels[result_i], gallerySaveNames[result_i], features[result_i]])

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

                            l2AllVal += l2Value
                        calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                    calTmp.sort(key=lambda x: x[3])
                    # centerImg.append([calTmp[4][0], calTmp[4][1], calTmp[4][2]])
                    centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])
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

                        l2AllVal += l2Value
                    calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                calTmp.sort(key=lambda x: x[3])
                # centerImg.append([calTmp[4][0], calTmp[4][1], calTmp[4][2]])
                centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])

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
        testPath = './' + datasetType + '\\' + 'testResult/' + modelType + '\\' + Fold + '/epochTermTest/'
        if not os.path.isdir(testPath):
            os.makedirs(testPath)
        with open(testPath + 'ReID_test_result_' + saveTestResultName + '.pickle',
                  'wb') as fw:
            pickle.dump(ReIDdict, fw)

    del allCamGalleryTestset
    del allCamGalleryLoader
    torch.cuda.empty_cache()