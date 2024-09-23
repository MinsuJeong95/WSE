import models.convnext as convnext
import models.convnextAttention as convnextAttention

import runConvNeXt
import runGCENet
import runSaveBestModel


def main(DBPath, Folds, datasetTypes):
    # Setting
    models = [convnext,
              convnextAttention]
    modelTypes = ['original',
                 'attention']

    #run
    runConvNeXt.run(datasetTypes, modelTypes, Folds, DBPath, models)
    runSaveBestModel.run(datasetTypes, modelTypes, Folds, DBPath, models)
    runGCENet.run(datasetTypes, modelTypes, Folds, DBPath, models)