import models.convnext as convnext
import models.convnextAttention as convnextAttention

import runConvNeXt
import runGCENet
import runSaveBestModel


def main():
    # Setting
    DBPath = 'D:\\JMS\\secondPaper\\thermalDB'
    models = [convnext,
              convnextAttention]
    Folds = ['Fold1',
            'Fold2']
    modelTypes = ['original',
                 'attention']
    datasetTypes = ['DBPerson-Recog-DB1_thermal',
                   'SYSU-MM01_thermal']

    #run
    runConvNeXt.run(datasetTypes, modelTypes, Folds, DBPath, models)
    runSaveBestModel.run(datasetTypes, modelTypes, Folds, DBPath, models)
    runGCENet.run(datasetTypes, modelTypes, Folds, DBPath, models)


main()