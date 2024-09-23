-----------------------------------------------------------------------------------------------------------------------------


## Storing the dataset

1. You can download DBPerson-Recog-DB1_thermal [1] :
```
[Click here to try](https://drive.google.com/file/d/1ugIeeHM0OTWhgNeF4ftP4AKE7s7ltYww/view?usp=sharing)
```
2. You can download SYSU-MM01 [2] :
```
https://github.com/wuancong/SYSU-MM01
```
3. The download path will be used to launch the code.


-----------------------------------------------------------------------------------------------------------------------------


## Instruction on launching code (Training, validation and test)

To launching WSE-Net, proceed as follows:

1. Download thermal dataset (DBPerson-Recog-DB1_thermal, SYSU-MM01).
2. Run WSE-Net_CLI on python :
```
python WSE-Net_CLI.py --dbpath=Your_dataset_path --fold=Fold1 or Fold2 or Fold1,Fold2 --datasettypes=DBPerson-Recog-DB1_thermal or SYSU-MM01_thermal or DBPerson-Recog-DB1_thermal,SYSU-MM01_thermal
```
3. After running, you can get the trained model, validation result and test result.


-----------------------------------------------------------------------------------------------------------------------------


## Download pre-trained models

```
https://drive.google.com/file/d/1BQ2spGZnMLj6RqvY2t5-dhG_o41bE810/view?usp=drive_link
```
If you need pre-trained models of WSE-Net, you can download it via the link above.


-----------------------------------------------------------------------------------------------------------------------------


## Prerequisites

- python 3.8.18 
- pytorch 1.11.0
- Windows 10


-----------------------------------------------------------------------------------------------------------------------------


## Reference


- [1] Nguyen, D.T.; Hong, H.G.; Kim, K.W.; Park, K.R. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 2017. 17(3): p. 605.

- [2] Wu, A.; Zheng, W.-S.; Gong, S.; Lai, J. RGB-IR person re-identification by cross-modality similarity preservation. Int. J. Comput. Vis., 2020, 128; pp. 1765-1785.
