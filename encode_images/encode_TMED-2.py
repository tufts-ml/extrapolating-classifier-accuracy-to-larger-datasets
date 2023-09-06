import argparse
import os
import re
import pandas as pd
from encoder import *
from utils import *
from torchvision.utils import save_image

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_TMED-2.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    folders = ['view_and_diagnosis_labeled_set', 'view_labeled_set']
    for folder in folders:
        directory = os.path.join(args.directory, '{}/labeled'.format(folder))
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if re.search('.png$', file):
                    path = os.path.join(subdir, file)
                    encoded_image = encode_image(model, path, device)
                    makedir_if_not_exist('{}/encoded_TMED-2/'.format(os.path.dirname(args.directory)))
                    encoded_path = '{}/encoded_TMED-2/{}.pt'.format(os.path.dirname(args.directory), re.split('\.', file)[0])
                    torch.save(encoded_image, encoded_path)
    keys = ['PLAX', 'PSAX', 'A4C', 'A2C', 'A4CorA2CorOther']
    values = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]]
    dictionary = {key: value for key, value in zip(keys, values)}
    labels_df = pd.read_csv(os.path.join(args.directory, 'labels_per_image.csv'))
    labels_df['study_id'] = labels_df.query_key.apply(lambda filename: re.split(r's', filename)[0])
    labels_df['label'] = labels_df.view_label.apply(lambda label: dictionary[label])
    labels_df['path'] = labels_df.query_key.apply(lambda filename: '{}/encoded_TMED-2/{}.pt'.format(os.path.dirname(args.directory), re.split('\.', filename)[0]))
    temp_df = labels_df[['study_id', 'label', 'path']].set_index('study_id')
    temp_df.to_csv(os.path.join('{}/encoded_TMED-2'.format(os.path.dirname(args.directory)), 'labels.csv'))