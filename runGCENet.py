import os

import GCENet.GCENet as GCENet
import GCENet.GCENetWithECA as GCENetWithECA
import GCENet.GCENetTraining as GCENetTraining
import GCENet.GCENetValidation as GCENetValidation
import GCENet.GCENetValidationAcc as GCENetValidationAcc
import GCENet.GCENetTest as GCENetTest

import calculate.APCalculate as APCalculate
import calculate.RankCalculate as RankCalculate

def run(datasetTypes, modelTypes, Folds, DBPath, models):
    for datasetType in datasetTypes:
        for Fold in Folds:
            modelType = 'GCE-Net'
            gceNet = GCENetWithECA

            trainDBPath = datasetType + '\\GCENetTrainData\\' + Fold + '\\train'
            valDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\validation\\' + 'allcam'
            testDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\test\\' + 'allcam'

            numOfClass = len(os.listdir(trainDBPath))
            splitChannel = int(768 / 2)

            #23-09-08JMS lr5->4 변경
            GCENetTraining.training(datasetType, Fold, modelType, gceNet, trainDBPath, numOfClass,
                                     numEpoch=30,
                                     startEpoch=0, lr=1e-4, wd=1e-6, splitChannel=splitChannel)
            GCENetValidation.validation(datasetType, Fold, modelTypes, modelType, models, gceNet, valDBPath,
                                         numOfClass, splitChannel=splitChannel)
            GCENetValidationAcc.accCalculate(datasetType, modelType, Fold)
            GCENetTest.test(datasetType, Fold, modelTypes, modelType, models, gceNet, testDBPath, numOfClass,
                             splitChannel=splitChannel)
            APCalculate.apCalculate(datasetType, modelType, Fold)
            RankCalculate.rankClaculate(datasetType, modelType, Fold)