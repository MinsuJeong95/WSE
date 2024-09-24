import click
import main

@click.command()
@click.option('--dbpath', prompt='Dataset path', help='Your dataset path.', type=str)
@click.option('--fold', prompt='Running fold', help='Can use Fold1 or Fold2 or Fold1, Fold2', type=str)
@click.option('--datasettypes', prompt='Type of dataset ', help='Can use '
                                                             'DBPerson-Recog-DB1_thermal or SYSU-MM01_thermal'
                                                             'or DBPerson-Recog-DB1_thermal, SYSU-MM01_thermal.',
              type=str)
def run(dbpath, fold, datasettypes):
    Path = dbpath
    datasetTypes = datasettypes.replace(" ", "").split(',')
    Folds = fold.replace(" ", "").split(',')

    main.main(Path, Folds, datasetTypes)


if __name__ == '__main__':
    run()