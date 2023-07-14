import argparse
import os
import re
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_Chest_X-Ray.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    for subdir, dirs, files in os.walk(args.directory):
        for file_index, file in enumerate(files):
            if re.search('.jpeg$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['PNEUMONIA', 'NORMAL'], 'Unexpected label: {}'.format(label)
                split = re.split('\/', subdir)[-2]
                assert split in ['train', 'test'], 'Unexpected split: {}'.format(split)
                path = os.path.join(subdir, file)
                encoded_image = encode_image(model, path, device)
                makedir_if_not_exist('{}/encoded_Chest_X-Ray/{}/{}/'.format(os.path.dirname(args.directory), split, label))
                encoded_path = '{}/encoded_Chest_X-Ray/{}/{}/{}.pt'.format(os.path.dirname(args.directory), split, label, re.split('\.', file)[0])
                torch.save(encoded_image, encoded_path)