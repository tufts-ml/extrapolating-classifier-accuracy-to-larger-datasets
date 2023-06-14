## Table of Contents

- [ChestX-ray8](#chestx-ray8)
- [Chest X-Ray](#chest-x-ray)
- [Breast Ultrasound Dataset](#breast-ultrasound-dataset)

## ChestX-ray8

1. `labels.csv` is generated when encoding ChestX-ray8 with `encode_ChestX-ray8.py`

## Chest X-Ray

1. Download and encode Chest X-Ray with `encode_Chest_X-Ray.py`.
2. Run `label_Chest_X-Ray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_Chest_X-Ray
           ├── test
           └── train
   ```

   run `label_Chest_X-Ray.py --directory='/Users/user/encode_Chest_X-Ray'`.

## Breast Ultrasound Dataset

1. Download and encode Breast Ultrasound Dataset with `encode_Breast_Ultrasound_Dataset.py`.
2. Run `label_Breast_Ultrasound_Dataset.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_Breast_Ultrasound_Dataset
           ├── benign
           ├── malignant
           └── normal
   ```

   run `label_Breast_Ultrasound_Dataset.py --directory='/Users/user/encoded_Breast_Ultrasound_Dataset'`.
   
