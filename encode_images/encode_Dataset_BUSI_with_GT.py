import argparse
import os
import re
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_Dataset_BUSI_with_GT.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    model = init_encoder()
    for subdir, dirs, files in os.walk(os.path.join(args.directory, 'Dataset_BUSI_with_GT/')):
        for file in files:
            if re.search('\([0-9]+\).png$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(subdir, file)
                encoded_image = encode_image(model, path)
                makedir_if_not_exist(os.path.join(args.directory, 'encoded_Dataset_BUSI_with_GT/{}/'.format(label)))
                encoded_path = os.path.join(args.directory, 'encoded_Dataset_BUSI_with_GT/{}/{}.pt'.format(label, re.split('\.', file)[0]))
                torch.save(encoded_image, encoded_path)