import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as nnf
import cv2
import os
import torch.nn.functional as F
import numpy as np

import pickle
import utils.imgPreprocess as imgPreprocess
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

def validation(datasetType, modelType, Fold, insertModel, DBPath, numOfClass):
    batchSize = 35

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)
    print("numOfClass :", numOfClass)

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    allCamGalleryValset = CustomDataset.CustomDataset(root_dir=DBPath,
                                                       transforms=trans)
    allCamGalleryLoader = DataLoader(allCamGalleryValset, batch_size=batchSize, shuffle=True, pin_memory=True)

    model = insertModel.convnext_small(pretrained=True, num_classes=numOfClass)  # convnext

    print('Fold : ' + Fold)

    totalLen = len(allCamGalleryValset)

    path = '.\\' + datasetType + '\\' + 'trainModels\\' + modelType + '\\' + Fold + '\epochTermModel'
    modelPaths = os.listdir(path)

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

    margin = 20.0
    cnt = 0
    for modelPath in modelPaths:
        ReIDdict = {}
        progress = 0
        model.load_state_dict(torch.load(path + '\\' + modelPath))
        model.eval()  # 평가모드
        model.to(device)

        cnt += 1
        if cnt % 15 == 0:
            margin -= 6
        criterionTriplet = nn.TripletMarginLoss(margin=margin).to(device)

        with torch.no_grad():
            ReIDresult = []
            keyTmp = 0

            for galleryI, galleryData in enumerate(allCamGalleryLoader):
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

                inputImgs = galleryImgs

                # 23-10-30 loss 수정확인
                outputs, tpFeature = model(inputImgs)
                anchor, positive, negative = choiceData(galleryLabels, tpFeature, device)
                tpLoss = criterionTriplet(anchor, positive, negative)
                loss = tpLoss

                progress = progress + outputs.shape[0]
                print("load : %.5f%%" % ((progress * 100) / totalLen))
                print("tploss : %.5f" % (tpLoss.item()))

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
        valResultPath = './' + datasetType + '\\' + 'valResult/' + modelType + '\\' + Fold + '\\' + '/epochTermValidation/'
        if not os.path.isdir(valResultPath):
            os.makedirs(valResultPath)
        with open(valResultPath + 'ReID_val_result' + '_' + saveValResultName + '.pickle',
                  'wb') as fw:
            pickle.dump(ReIDdict, fw)

    del allCamGalleryValset
    del allCamGalleryLoader
    torch.cuda.empty_cache()
