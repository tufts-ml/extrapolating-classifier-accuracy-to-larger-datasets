## Table of Contents

- [ChestX-ray8](#chestx-ray8)
- [Chest X-Ray](#chest_x-ray)
- [Breast Ultrasound Dataset](#breast-ultrasound-dataset)

## ChestX-ray8

1. Consolidate downloaded [images](https://nihcc.app.box.com/v/ChestXray-NIHCC) and `Data_Entry_2017_v2020.csv` into a folder named `ChestX-ray8`.
2. Run `encode_ChestX-ray8.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── ChestX-ray8
           ├── 00000001_000.png
           ├── 00000001_001.png
           ├── ...
           ├── ...
           ├── ...
           ├── 00030805_000.png
           └── Data_Entry_2017_v2020.csv
   ```

   run `encode_ChestX-ray8.py --directory='/Users/user/'`.
   
## PneumoniaMNIST

1. Download [ChestXRay2017.zip](https://data.mendeley.com/datasets/rscbjbr9sj/2) into a folder named `Chest X-Ray`.
2. Run `encode_Chest_X-Ray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── Chest_X-Ray
           ├── test
           └── train
   ```

   run `encode_Chest_X-Ray.py --directory='/Users/user/'`.

## Breast Ultrasound Dataset

1. Download [Dataset_BUSI.zip](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) into a folder named `Breast Ultrasound Dataset`.
2. Run `encode_Breast_Ultrasound_Dataset.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── Breast_Ultrasound_Dataset
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_Breast_Ultrasound_Dataset.py --directory='/Users/user/'`.
   
