## Table of Contents

- [ChestX-ray14](#chestx-ray14)
- [Chest X-Ray](#chest-x-ray)
- [BUSI](#busi)
- [OASIS-3](#oasis-3)

## ChestX-ray8

1. Consolidate downloaded [images](https://nihcc.app.box.com/v/ChestXray-NIHCC) and [Data_Entry_2017_v2020.csv](https://nihcc.app.box.com/v/ChestXray-NIHCC) into a folder named `ChestX-ray8`.
2. Run `encode_ChestX-ray14.py --directory='{path_to_dataset}'`.

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

   run `encode_ChestX-ray14.py --directory='/Users/user/ChestX-ray14'`.
   
## Chest X-Ray

1. Download [ChestXRay2017.zip](https://data.mendeley.com/datasets/rscbjbr9sj/2) into a folder named `Chest X-Ray`.
2. Run `encode_Chest_X-Ray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── Chest_X-Ray
           ├── test
           └── train
   ```

   run `encode_Chest_X-Ray.py --directory='/Users/user/Chest_X-Ray'`.

## BUSI

1. Download [Dataset_BUSI.zip](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) into a folder named `BUSI`.
2. Run `encode_BUSI.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── BUSI
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_BUSI.py --directory='/Users/user/BUSI'`.   

## OASIS-3

1. See [https://github.com/NrgXnat/oasis-scripts](https://github.com/NrgXnat/oasis-scripts) for downloading CT scans.
2. Run `encode_OASIS-3.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── OASIS-3
           ├── OAS30001_CT_d2438
           ├── OAS30001_CT_d3132
           ├── ...
           ├── ...
           ├── ...
           ├── OAS31473_CT_d0126
           ├── ADRC_Clinical_Data.csv
           └── CT_Sessions.csv
   ```

   run `encode_OASIS-3.py --directory='/Users/user/OASIS-3'`.   
