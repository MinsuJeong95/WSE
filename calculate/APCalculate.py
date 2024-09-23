import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from collections import defaultdict

import shutil

def apCalculate(datasetType, modelType, Fold):
    path = './' + datasetType + '\\' + 'testResult' + '\\' + modelType + '\\' + Fold
    pickleFilePaths = os.listdir(path + '/epochTermTest')

    print(pickleFilePaths)

    for pickleFilePath in pickleFilePaths:
        folderPath = path + '/testResultGraph' + '/' + pickleFilePath.split('.')[0] + '/apCalculate'
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error')

        m = nn.Softmax(dim=1)
        mAP = 0


        print('Fold : ' + Fold)
        print('modelType : ' + modelType)

        with open(path + '/epochTermTest' + '/' + pickleFilePath, 'rb') as fr:
            loadReIDdict = pickle.load(fr)
        ReIDdict = loadReIDdict

        for i, (key, value) in enumerate(ReIDdict.items()):

            classPath = folderPath + '/' + str(key)
            try:
                if not os.path.exists(classPath):
                    os.makedirs(classPath)
            except OSError:
                print('Error')

            graphPath = classPath + '/PRCurve'
            try:
                if not os.path.exists(graphPath):
                    os.makedirs(graphPath)
            except OSError:
                print('Error')

            numpyPath = classPath + '/PRdatas'
            try:
                if not os.path.exists(numpyPath):
                    os.makedirs(numpyPath)
            except OSError:
                print('Error')

            allReIdValues = []

            TP = 0
            FP = 0
            Recall = []
            Precision = []
            trueLabelCnt = 0

            for i, ReIDvalue in enumerate(value):
                label = ReIDvalue[0]
                if label == key:
                    trueLabelCnt = trueLabelCnt + 1

            for t, ReIDvalue in enumerate(value):
                label = ReIDvalue[0]
                if label == key:
                    TP = TP + 1
                elif label != key:
                    FP = FP + 1

                if TP + FP == 0:
                    Precision.append(0)
                else:
                    Precision.append(TP / (TP + FP))

                if trueLabelCnt == 0:
                    Recall.append(0)
                else:
                    Recall.append(TP / trueLabelCnt)

            apPrecision = Precision
            apRecall = Recall

            plt.figure()
            plt.axis([min(apRecall), max(apRecall), 0, 1])
            plt.grid(True)
            plt.plot(apRecall, apPrecision)

            AP = 0
            for apRecall_i in range(len(apRecall)):
                if apRecall_i == 0:
                    AP = AP + (apPrecision[apRecall_i] * apRecall[apRecall_i])
                else:
                    AP = AP + (apPrecision[apRecall_i] * (apRecall[apRecall_i] - apRecall[apRecall_i-1]))



            mAP = mAP + AP

            AP = int(AP * (10000))
            AP = float(AP) / (100)


            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(
                'PRCurve_' + Fold + '_' + modelType + '_' + str(key) + ' AP : ' + str(AP))
            plt.fill_between(apRecall[0:len(apRecall)], apPrecision[0:len(apPrecision)], alpha=0.2)

            plt.savefig(
                graphPath + '/PRCurve_' + Fold + '_' + modelType + '_' + str(key) + '_' + str(
                    AP) + '.png')
            np.save(numpyPath + '/apRecall_' + Fold + '_' + modelType + '_' + str(key),
                    np.array(apRecall))
            np.save(numpyPath + '/apPrecision_' + Fold + '_' + modelType + '_' + str(key),
                    np.array(apPrecision))

            plt.close()

        f = open(folderPath + "/mAP" + ".txt", 'w')
        f.write(pickleFilePath.split('.')[0] + '\n')
        mAP = mAP / len(ReIDdict)
        f.write('mAP : ' + str(mAP) + '\n')
        f.close()
