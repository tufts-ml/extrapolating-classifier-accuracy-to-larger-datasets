import argparse
import os
import re
import pandas as pd
import torch
from torchvision.utils import save_image
# Importing our custom module(s)
import encoder
import utils

def label_oasis(path):
    scan_df = pd.read_csv(os.path.join(path, 'CT_Sessions.csv'))
    scan_df['scan_day'] = scan_df['XNAT_CTSESSIONDATA ID'].apply(lambda item: int(re.split('\_', item)[-1][1:]))
    diagnosis_df = pd.read_csv(os.path.join(path, 'ADRC_Clinical_Data.csv'))
    diagnosis_df['diagnosis_day'] = diagnosis_df['ADRC_ADRCCLINICALDATA ID'].apply(lambda item: int(re.split('\_', item)[-1][1:]))
    df = pd.merge(scan_df, diagnosis_df, on='Subject', how='inner')
    df['diff'] = df['scan_day'] - df['diagnosis_day']
    df['abs_diff'] = abs(df['scan_day'] - df['diagnosis_day'])
    df = df[(df['diff']<=80)&(df['diff']>=-360)]
    df = df.loc[df.groupby(['XNAT_CTSESSIONDATA ID'])['abs_diff'].idxmax()]
    df = df.rename(columns={'Subject': 'study_id'})
    df['label'] = df['cdr'].apply(lambda item: [0] if item == 0.0 else [1])
    df['path'] = df['XNAT_CTSESSIONDATA ID'].apply(lambda item: '{}/encoded_OASIS-3_CT/{}.pt'.format(os.path.dirname(path), item))
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_OASIS-3_CT.py')
    parser.add_argument('--dataset_path', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    
    utils.makedir_if_not_exist('{}/encoded_OASIS-3_CT'.format(os.path.dirname(args.dataset_path)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = encoder.init_encoder()
    model.to(device)
    
    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if re.search('\_CT.nii.gz$', file):
                encoded_image = encoder.encode_scan(model, os.path.join(root, file))
                encoded_path = '{}/encoded_OASIS-3_CT/{}.pt'.format(os.path.dirname(args.dataset_path), re.split('\/', subdir)[-6])
                count = 1
                while os.path.isfile(encoded_path):
                    encoded_path = '{}/encoded_OASIS-3_CT/{} ({}).pt'.format(os.path.dirname(args.dataset_path), re.split('\/', subdir)[-6], count)
                    count += 1
                torch.save(encoded_image, encoded_path)
                
    df = label_oasis(args.dataset_path)
    df = df[['study_id', 'label', 'path']].set_index('study_id')
    df.to_csv('{}/encoded_OASIS-3_CT/labels.csv'.format(os.path.dirname(args.dataset_path)))