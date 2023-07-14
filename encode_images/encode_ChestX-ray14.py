import argparse
import os
import re
import pandas as pd
import numpy as np
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_ChestX-ray14.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    for subdir, dirs, files in os.walk(args.directory):
        for file in files:
            if re.search('[0-9]{8}\_[0-9]{3}.png$', file):
                path = os.path.join(subdir, file)
                encoded_image = encode_image(model, path, device)
                makedir_if_not_exist(os.path.join(os.path.dirname(args.directory), 'encoded_images/'))
                encoded_path = os.path.join(os.path.dirname(args.directory), 'encoded_images/{}.pt'.format(re.split('\.', file)[0]))
                torch.save(encoded_image, encoded_path)
    df = pd.read_csv(os.path.join(args.directory, 'Data_Entry_2017_v2020.csv'))
    encoded_directory = os.path.join(os.path.dirname(args.directory), 'encoded_images/')
    df['study_id'] = df['Patient ID']
    #labels = np.unique(np.concatenate([re.split('\|', findings) for findings in df['Finding Labels'].to_list()]))
    labels = ['Atelectasis', 'Effusion', 'Infiltration']
    df['label'] = df['Finding Labels'].apply(lambda findings: one_hot_encode_list(findings, labels))
    df['path'] = df['Image Index'].apply(lambda file: os.path.join(os.path.dirname(args.directory), 'encoded_images/{}.pt'.format(re.split('\.', file)[0])))
    temp_df = df[['study_id', 'label', 'path']].set_index('study_id')
    temp_df.to_csv(os.path.join(encoded_directory, 'labels.csv'))