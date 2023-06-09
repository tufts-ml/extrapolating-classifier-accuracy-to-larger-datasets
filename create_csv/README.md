## Table of Contents

- [ChestMNIST](#chestmnist)
- [PneumoniaMNIST](#pneumoniamnist)
- [BrestMNIST](#brestmnist)

## ChestMNIST

1. Download and econde PneumoniaMNIST with `encode_ChestXray-NIHCC.py`
2. Run `encode_ChestXray-NIHCC.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_ChestXray-NIHCC
           ├── Data_Entry_2017_v2020.csv
           ├── 00000001_000.png
           ├── 00000001_001.png
           ├── 00000001_002.png
           ├── ...
           ├── ...
           └── ...
   ```

   run `encode_ChestXray-NIHCC.py --directory='/Users/user/encoded_ChestXray-NIHCC/'`.

## PneumoniaMNIST

1. Download and econde PneumoniaMNIST with `encode_chest_xray.py`.
2. Run `create_csv_chest_xray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_chest_xray
           ├── test
           └── train
   ```

   run `encode_chest_xray.py --directory='/Users/user/encoded_chest_xray/'`.

## BrestMNIST

1. Download and econde BrestMNIST with `create_csv_Dataset_BUSI_with_GT.py`.
2. Run `create_csv_Dataset_BUSI_with_GT.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── encoded_Dataset_BUSI_with_GT
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_Dataset_BUSI_with_GT.py --directory='/Users/user/encoded_Dataset_BUSI_with_GT/'`.
   
