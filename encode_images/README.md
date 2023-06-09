## Table of Contents

- [PneumoniaMNIST](#pneumoniamnist)
- [BrestMNIST](#brestmnist)

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
   