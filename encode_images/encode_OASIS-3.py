import argparse
import os
import re
from encoder import *
from utils import *
from torchvision.utils import save_image
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='encode_OASIS-3.py')
    parser.add_argument('--directory', type=str, help='Directory where images are saved', required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_encoder()
    model.to(device)
    for subdir, dirs, files in os.walk(args.directory):
        for file in files:
            if re.search('\_CT.nii.gz$', file):
                path = os.path.join(subdir, file)
                encoded_image = encode_scan(model, device, path)
                makedir_if_not_exist('{}/encoded_OASIS-3/'.format(os.path.dirname(args.directory)))
                encoded_path = '{}/encoded_OASIS-3/{}.pt'.format(os.path.dirname(args.directory), re.split('\/', subdir)[-6])
                count = 1
                while os.path.isfile(encoded_path):
                    encoded_path = '{}/encoded_OASIS-3/{} ({}).pt'.format(os.path.dirname(args.directory), re.split('\/', subdir)[-6], count)
                    count += 1
                torch.save(encoded_image, encoded_path)
    df1 = pd.read_csv(os.path.join(args.directory, 'CT_Sessions.csv'))[['XNAT_CTSESSIONDATA ID', 'Subject']]
    df1['df1_day'] = df1['XNAT_CTSESSIONDATA ID'].apply(lambda x: int(re.split('\_', x)[-1][1:]))
    df2 = pd.read_csv(os.path.join(args.directory, 'ADRC_Clinical_Data.csv'))[['ADRC_ADRCCLINICALDATA ID', 'Subject', 'cdr']]
    df2['df2_day'] = df2['ADRC_ADRCCLINICALDATA ID'].apply(lambda x: int(re.split('\_', x)[-1][1:]))
    merged_df = pd.merge(df1, df2, on='Subject', how='inner')
    merged_df['diff'] = abs(merged_df['df1_day'] - merged_df['df2_day'])
    merged_df = merged_df.loc[merged_df.groupby(['Subject', 'df1_day'])['diff'].idxmin()]
    merged_df['diff'] = abs(merged_df['df2_day'] - merged_df['df1_day'])
    merged_df = merged_df[(merged_df['diff']>=0)&(merged_df['diff']<=365)]
    merged_df = merged_df.rename(columns={'Subject': 'study_id'})
    merged_df['label'] = merged_df['cdr'].apply(lambda x: [0] if x == 0.0 else [1])
    merged_df['path'] = merged_df['XNAT_CTSESSIONDATA ID'].apply(lambda x: '{}/encoded_OASIS-3/{}.pt'.format(os.path.dirname(args.directory), x))
    temp_df = merged_df[['study_id', 'label', 'path']].set_index('study_id')
    temp_df.to_csv('{}/encoded_OASIS-3/labels.csv'.format(os.path.dirname(args.directory)))