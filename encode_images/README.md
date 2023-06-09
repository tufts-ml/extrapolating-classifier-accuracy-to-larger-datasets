## Table of Contents

- [ChestMNIST](#chestmnist)
- [PneumoniaMNIST](#pneumoniamnist)
- [BrestMNIST](#brestmnist)

## ChestMNIST

1. Consolidate downloaded [images](https://nihcc.app.box.com/v/ChestXray-NIHCC) into a folder named `ChestXray-NIHCC`.
2. Run `encode-chestxray-nihcc.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── ChestXray-NIHCC
           ├── 00000001_000.png
           ├── 00000001_001.png
           ├── 00000001_002.png
           ├── ...
           ├── ...
           └── ...
   ```

   run `encode-chestxray-nihcc.py --directory='/Users/user/'`.
   
## PneumoniaMNIST

1. Download [ChestXRay2017.zip](https://data.mendeley.com/datasets/rscbjbr9sj/2).
2. Run `encode_chest_xray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── chest_xray
           ├── test
           └── train
   ```

   run `encode-chest-xray.py --directory='/Users/user/'`.

## BrestMNIST

1. Download [Dataset_BUSI.zip](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset).
2. Run `encode_Dataset_BUSI_with_GT.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/`

   ```
   └── user
       └── Dataset_BUSI_with_GT
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_Dataset_BUSI_with_GT.py --directory='/Users/user/'`.
   
