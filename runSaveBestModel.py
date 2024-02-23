import os
import saveBestModel.saveBestModel as saveBestModel
import saveBestModel.saveFeatureMap as saveFeatureMap
import saveBestModel.saveFeatureMapWithPatchImage as saveFeatureMapWithPatchImage

def run(datasetTypes, modelTypes, Folds, DBPath, models):
    saveBestModel.saveBestModel(Folds, modelTypes, datasetTypes)
    for datasetType in datasetTypes:
        for Fold in Folds:
            trainDBPath = DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\allcam'
            numOfClass = len(os.listdir(trainDBPath))
            # Save SCE-Net Training Dataset
            if os.path.isdir('.\\' + datasetType + '\\GCENetTrainData\\' + Fold + '\\train\\'):
                continue
            saveFeatureMapWithPatchImage.saveFeatureMap(datasetType, modelTypes, Fold, models, DBPath, numOfClass)
            # saveFeatureMap.saveFeatureMap(datasetType, modelTypes, Fold, models, DBPath, numOfClass)