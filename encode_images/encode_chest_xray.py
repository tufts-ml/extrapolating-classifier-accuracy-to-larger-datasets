import argparse
import os
import re
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_chest_xray.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    model = init_encoder()
    for subdir, dirs, files in os.walk(os.path.join(args.directory, 'chest_xray/')):
        for file_index, file in enumerate(files):
            if re.search('.jpeg$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['PNEUMONIA', 'NORMAL'], 'Unexpected label: {}'.format(label)
                split = re.split('\/', subdir)[-2]
                assert split in ['train', 'test'], 'Unexpected split: {}'.format(split)
                path = os.path.join(subdir, file)
                encoded_image = encode_image(model, path)
                makedir_if_not_exist(os.path.join(args.directory, 'encoded_chest_xray/{}/{}/'.format(split, label)))
                encoded_path = os.path.join(args.directory, 'encoded_chest_xray/{}/{}/{}.pt'.format(split, label, re.split('\.', file)[0]))
                torch.save(encoded_image, encoded_path)