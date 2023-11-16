import argparse
import os
import re
import torch
from torchvision.utils import save_image
# Importing our custom module(s)
import encoder
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_BUSI.py')
    parser.add_argument('--dataset_path', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = encoder.init_encoder()
    model.to(device)
    
    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if re.search('\([0-9]+\).png$', file):
                label = re.split('\/', root)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(root, file)
                encoded_image = encoder.encode_image(model, path, grayscale=False)
                utils.makedir_if_not_exist('{}/encoded_BUSI/{}/'.format(os.path.dirname(args.dataset_path), label))
                encoded_path = '{}/encoded_BUSI/{}/{}.pt'.format(os.path.dirname(args.dataset_path), label, re.split('\.', file)[0])
                torch.save(encoded_image, encoded_path)