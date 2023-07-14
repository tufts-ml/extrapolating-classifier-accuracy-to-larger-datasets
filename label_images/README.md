## Table of Contents

- [ChestX-ray14](#chestx-ray14)
- [Chest X-Ray](#chest-x-ray)
- [BUSI](#busi)
- [OASIS-3](#oasis-3)

## ChestX-ray14

1. `labels.csv` is generated when encoding ChestX-ray14 with `encode_ChestX-ray14.py`

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

## BUSI

1. Download and encode Breast Ultrasound Dataset with `encode_BUSI.py`.
2. Run `label_BUSI.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_BUSI
           ├── benign
           ├── malignant
           └── normal
   ```

   run `label_BUSI.py --directory='/Users/user/encoded_BUSI'`.
   
## OASIS-3

1. `labels.csv` is generated when encoding OASIS-3 with `encode_OASIS-3.py`
