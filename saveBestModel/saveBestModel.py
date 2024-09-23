import os
from collections import defaultdict
import shutil

def saveBestModel(Folds, modelTypes, datasetTypes):
    for datasetType in datasetTypes:
        for modelType in modelTypes:
            #Save best model
            for Fold in Folds:
                loadPath = './' + datasetType + '\\' + 'valResult' + '\\' + modelType + '\\' + Fold + '\\selectEpoch'
                loadModel = os.listdir(loadPath)
                savePath = './' + datasetType + '\\' + 'bestModel' + '\\' + modelType + '\\' + Fold
                if not os.path.isdir(savePath):
                    os.makedirs(savePath)
                shutil.copyfile(loadPath + '\\' + loadModel[0], savePath + '\\' + loadModel[0])

