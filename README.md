-----------------------------------------------------------------------------------------------------------------------------
## Weak Saliency Ensemble Network for Person Re-identification Using Infrared Light Images

This study proposes a new weak saliency ensemble network (WSE-Net) for person re-id using IR images. WSE-Net incorporates a channel reduction of feature (CRF) method to reduce computational cost in the ensemble network, a technique for converting input images into group of patch images and feeding them into the ensemble model to enhance the reduced feature information, and a grouped convolution ensemble network (GCE-Net) that enables the fusion of features extracted from original and attention-guided ensemble models.

-----------------------------------------------------------------------------------------------------------------------------


## Storing the dataset

1. You can download DBPerson-Recog-DB1_thermal [1] :
   
   <https://github.com/MinsuJeong95/OADE>

2. You can download SYSU-MM01 [2] :
   
   <https://github.com/wuancong/SYSU-MM01>


3. The download path will be used to launch the code.


-----------------------------------------------------------------------------------------------------------------------------
## Download Requirements.txt and WSE-Net_CLI (Command-Line Interface)
1. Requirements.txt

   <https://drive.google.com/file/d/1GUCcXzesZ6rXPKc2SWw8eSHrZmIFCAbD/view?usp=drive_link>
   
   We used virtual environment (Anaconda 4.9.2).
   
   You can install our environment :
   
    ```
    conda create --name WSE-Net --file requirements.txt
    conda activate WSE-Net
    conda install conda-forge::timm
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```
   * For pytorch, you should install it for your CUDA version.



3. WSE-Net_CLI to interact with our code.

   <https://drive.google.com/file/d/1zkl0Smfq98c_xZgeJG-kLfOPQFNqwWxf/view?usp=drive_link>

-----------------------------------------------------------------------------------------------------------------------------



## Instruction on launching code (Training, validation and test)

To launching WSE-Net, proceed as follows:

1. Download thermal dataset (DBPerson-Recog-DB1_thermal, SYSU-MM01).
2. Run WSE-Net_CLI on python.
   
   2.1 Run DBPerson-Recog-DB1_thermal.
   ```
   python WSE-Net_CLI.py --dbpath=Your_dataset_path --fold=Fold1,Fold2 --datasettypes=DBPerson-Recog-DB1_thermal
   ```
   2.2 Run SYSU-MM01_thermal.
   ```
   python WSE-Net_CLI.py --dbpath=Your_dataset_path --fold=Fold1,Fold2 --datasettypes=SYSU-MM01_thermal
   ```
3. After running, you can get the trained model, validation result and test result.


-----------------------------------------------------------------------------------------------------------------------------


## Download pre-trained models

   <https://drive.google.com/file/d/1BQ2spGZnMLj6RqvY2t5-dhG_o41bE810/view?usp=drive_link>

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
