import os
import Training
import Validation
import validationAcc
import Test
import calculate.APCalculate as APCalculate
import calculate.RankCalculate as RankCalculate


def run(datasetTypes, modelTypes, Folds, DBPath, models):
    for datasetType in datasetTypes:
        for model_i, modelType in enumerate(modelTypes):
            for Fold in Folds:
                trainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\allcam'
                valDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\validation\\allcam'
                testDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\test\\allcam'

                numOfClass = len(os.listdir(trainDBPath))

                Training.training(datasetType, modelType, Fold, models[model_i], trainDBPath, numOfClass,
                                  numEpoch=30, startEpoch=0, lr=1e-4, wd=1e-5)
                Validation.validation(datasetType, modelType, Fold, models[model_i], valDBPath, numOfClass)
                validationAcc.accCalculate(datasetType, modelType, Fold)
                Test.test(datasetType, modelType, Fold, models[model_i], testDBPath, numOfClass)
                APCalculate.apCalculate(datasetType, modelType, Fold)
                RankCalculate.rankClaculate(datasetType, modelType, Fold)
