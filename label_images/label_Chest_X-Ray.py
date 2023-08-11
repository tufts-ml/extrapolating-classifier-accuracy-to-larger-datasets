import argparse
import os
import re
import numpy as np
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='label_Chest_X-Ray.py')
    parser.add_argument('--directory', type=str, help='Directory where encoded images are saved', required=True)
    args = parser.parse_args()
    file_index = 0
    folder_df = pd.DataFrame(columns=['label', 'split', 'person_id', 'path'])
    for subdir, dirs, files in os.walk(args.directory):
        for file in files:
            if re.search('.pt$', file):
                label = re.split('\/', subdir)[-1]
                assert label in ['PNEUMONIA', 'NORMAL'], 'Unexpected label: {}'.format(label)
                split = re.split('\/', subdir)[-2]
                assert split in ['train', 'test'], 'Unexpected split: {}'.format(split)
                path = os.path.join(subdir, file)
                # File starts with 'IM'
                if re.search('^IM', file):
                    person_id = re.split('\-', file)[1]
                # File starts with 'Normal'
                elif re.search('^Normal', file) or re.search('^NORMAL2', file):
                    person_id = re.split('\-', file)[2]
                # File starts with 'Person'
                elif re.search('^person', file):
                    person_id = re.sub('^person', '', re.split('\_', file)[0])
                    label = re.split('\_', file)[1]
                else:
                    assert False, 'Unexpected file: {}'.format(file)
                folder_df.loc[file_index] = [label, split, person_id, path]
                file_index += 1
    # Sort df and add 'study_id' column
    folder_df['person_id'] = pd.to_numeric(folder_df['person_id'])
    folder_df = folder_df.sort_values(['label', 'split', 'person_id'])
    mask = folder_df.duplicated(['label', 'split', 'person_id'])
    study_id = mask.copy()
    study_id[~mask] = np.arange(sum(~mask))
    study_id[mask] = np.nan
    folder_df['study_id'] = study_id
    folder_df.study_id = folder_df.study_id.ffill()
    # Write .csv to directory
    temp_df = folder_df[['study_id', 'label', 'path']].set_index('study_id')
    temp_df.label = temp_df.label.apply(lambda label: [0, 0] if label == 'NORMAL' else [1, 0] if label == 'bacteria' else [0, 1])
    temp_df.to_csv(os.path.join(args.directory, 'labels.csv'))