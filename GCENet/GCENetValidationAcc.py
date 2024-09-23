import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import shutil
from collections import defaultdict


def mySoftmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def findTopFeature(features):
    distenceResult = []
    camNames = []
    result = 0
    resultImg = 0
    resultName = 0
    resultAll = 0

    # Calculate Images Center
    for center in range(len(features)):
        resultAll = 0
        for f_i in range(len(features)):
            # result = 0
            if center == f_i:
                continue
            # distence = (features[center] - features[f_i]) * (features[center] - features[f_i])  # L2 distance
            # for d_i in range(len(distence)):
            #     result = result + distence[d_i]
            # result = result ** (1 / 2)
            result = torch.dist(features[center], features[f_i])
            resultAll += result
        resultAll /= (len(features)-1)

        distenceResult.append(resultAll)
    pickImg = min(distenceResult)
    for d_i in range(len(distenceResult)):
        if pickImg == distenceResult[d_i]:
            resultImg = features[d_i]
            break

    return resultImg


def accCalculate(datasetType, modelType, Fold):
    path = '.\\' + datasetType + '\\valResult\\' + modelType + '\\' + Fold
    pickleFilePath = os.listdir(path + '\\epochTermValidation')

    # 피클list
    modelNum = []
    for i in range(len(pickleFilePath)):  # 피클list 재배열
        modelName = pickleFilePath[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = pickleFilePath[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pickle')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        pickleFilePath[num_i] = saveName

    print(len(pickleFilePath))

    Accuracy = []
    lossArray = []
    for fileCnt in range(len(pickleFilePath)):
        print(fileCnt)
        with open(path + '\\epochTermValidation'+'/'+pickleFilePath[fileCnt], 'rb') as fr:
            loadReIDdict = pickle.load(fr)
        ReIDdict = loadReIDdict

        folderPath = path + '\\valResultGraph' + '/' + pickleFilePath[fileCnt].split('.')[0]
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error')

        f = open(folderPath+"/uncorrected" + ".txt", 'w')

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        Recall = 0
        Precision = 0
        totalLoss = 0
        totalScore = 0
        valueCnt = 0
        allReIDValue = []
        allFeatureValues = []
        allLabels = []
        distanceFeature = defaultdict(list)
        accTmp = 0

        for i, (key, value) in enumerate(ReIDdict.items()):
            if key == 0:
                continue
            for valueData in value:
                allFeatureValues.append(valueData[0])
                allLabels.append(key)

                totalLoss += valueData[1]
                valueCnt += 1

        for i, (key, value) in enumerate(ReIDdict.items()):
            features = []
            if key == 0:
                continue
            print(key)
            for valueData in value:
                features.append(valueData[0])
            topFeature = findTopFeature(features)

            for feature_i, ReIDValue in enumerate(allFeatureValues):
                result = torch.dist(topFeature, ReIDValue, 2)
                if result == 0:
                    continue
                distanceFeature[key].append((result, allLabels[feature_i]))

        for i, (key, values) in enumerate(distanceFeature.items()):
            TP = 0
            FP = 0
            values.sort(key=lambda x: x[0])
            for value in values:
                if key == value[1]:
                    TP = TP + 1
                elif key != value[1]:
                    FP = FP + 1

                if TP >= (len(ReIDdict[key])-1):
                    break
            accTmp = accTmp + ((len(values)-FP) / len(values))


        acc = accTmp / len(distanceFeature)
        Accuracy.append(acc)
        print('Accuracy : ', acc)
        f.write('Accuracy : ' + str(acc) + '\n')

        loss = totalLoss / valueCnt
        lossArray.append(loss)
        print('Loss : ', loss)
        f.write('Loss : ' + str(loss) + '\n')

        plt.close()
        f.close()

    plt.figure()
    plt.axis([1, len(pickleFilePath), 0, 1])
    plt.grid(True)
    plt.plot(range(1, len(pickleFilePath) + 1), Accuracy, label='Accuracy')
    plt.plot(range(1, len(pickleFilePath) + 1), lossArray, label='Loss')

    foldType = Fold

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy_Loss')
    plt.title('Validation_' + foldType + '_' + modelType)
    plt.legend(loc='center right', ncol=1)

    folderPath = '.\\' + datasetType + '\\valResult\\' + modelType + '\\' + Fold + '\\valResultGraph'
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)
    plt.savefig(folderPath + '/Val_' + foldType + '_' + modelType + '.png')
    plt.close()

    np.save(folderPath + '/Val_acc_' + foldType + '_' + modelType, np.array(Accuracy))
    np.save(folderPath + '/Val_loss_' + foldType + '_' + modelType, np.array(lossArray))

    # 모델 select
    accMax = max(Accuracy)
    print(accMax)
    modelSelect = []
    for i in range(len(Accuracy)):
        if Accuracy[i] == accMax:
            modelSelect.append(i)
    trainPath = '.\\' + datasetType + '\\trainModels\\' + modelType + '\\' + Fold + '\epochTermModel'
    modelPath = os.listdir(trainPath)

    # 모델list
    modelNum = []
    for i in range(len(modelPath)):  # 모델list 재배열
        modelName = modelPath[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = modelPath[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pth')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        modelPath[num_i] = saveName
    selectTrainPath = path + '\\selectEpoch\\'
    if not os.path.isdir(selectTrainPath):
        os.makedirs(selectTrainPath)
    for modelSelect_num in modelSelect:
        shutil.copyfile(trainPath + '\\' + modelPath[modelSelect_num], selectTrainPath + modelPath[modelSelect_num])
