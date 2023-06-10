## Table of Contents

- [ChestX-ray8](#chestx-ray8)
- [Chest X-Ray](#chest-x-ray)
- [Breast Ultrasound Dataset](#breast-ultrasound-dataset)

## ChestX-ray8

1. Download and encode ChestX-ray8 with `encode_ChestX-ray8.py`
2. Run `create_csv_ChestX-ray8.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_ChestX-ray8
           ├── 00000001_000.png
           ├── 00000001_001.png
           ├── ...
           ├── ...
           ├── ...
           ├── 00030805_000.png
           └── Data_Entry_2017_v2020.csv
   ```

   run `encode_ChestX-ray8.py --directory='/Users/user/encoded_ChestX-ray8'`.

## Chest X-Ray

1. Download and encode Chest X-Ray with `encode_Chest_X-Ray.py`.
2. Run `create_csv_Chest_X-Ray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_Chest_X-Ray
           ├── test
           └── train
   ```

   run `encode_Chest_X-Ray.py --directory='/Users/user/encoded_chest_xray'`.

## Breast Ultrasound Dataset

1. Download and encode Breast_Ultrasound_Dataset with `encode_Breast_Ultrasound_Dataset.py`.
2. Run `create_csv_Breast_Ultrasound_Dataset.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_Breast_Ultrasound_Dataset
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_Breast_Ultrasound_Dataset.py --directory='/Users/user/encoded_Breast_Ultrasound_Dataset'`.
   
