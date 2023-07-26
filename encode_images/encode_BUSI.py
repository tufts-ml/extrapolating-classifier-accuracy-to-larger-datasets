import argparse
import os
import re
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_BUSI.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    for subdir, dirs, files in os.walk(args.directory):
        for file in files:
            if re.search('\([0-9]+\).png$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['benign', 'malignant', 'normal'], 'Unexpected label: {}'.format(label)
                path = os.path.join(subdir, file)
                encoded_image = encode_image(model, path, device, grayscale=False)
                makedir_if_not_exist('{}/encoded_BUSI/{}/'.format(os.path.dirname(args.directory), label))
                encoded_path = '{}/encoded_BUSI/{}/{}.pt'.format(os.path.dirname(args.directory), label, re.split('\.', file)[0])
                torch.save(encoded_image, encoded_path)