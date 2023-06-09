## Table of Contents

- [PneumoniaMNIST](#pneumoniamnist)
- [BrestMNIST](#brestmnist)

## PneumoniaMNIST

1. Download and econde PneumoniaMNIST with `encode_chest_xray.py`.

2. Run `create_csv_chest_xray.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/encoded_chest_xray/`

   ```
   └── user
       └── encoded_chest_xray
           ├── test
           └── train
   ```

   run `encode-chest-xray.py --directory='/Users/user/'`.

## BrestMNIST

1. Download and econde BrestMNIST with `create_csv_Dataset_BUSI_with_GT.py`.

2. Run `create_csv_Dataset_BUSI_with_GT.py --directory='{path_to_dataset}'`.

   For example, if the dataset is in `/Users/user/encoded_Dataset_BUSI_with_GT/`

   ```
   └── user
       └── encoded_Dataset_BUSI_with_GT
           ├── benign
           ├── malignant
           └── normal
   ```

   run `encode_Dataset_BUSI_with_GT.py --directory='/Users/user/'`.
   