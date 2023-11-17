
import dask.dataframe as dd
import datetime as dt
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from pathlib import Path




def add_image_note(row):
    prefix = 'p' + str(row['subject_id'])[:2]
    pat = 'p' + str(int(row['subject_id']))
    study = 's'+ str(int(row['study_id']))
    pat_path = os.path.join(mimic_cxr_note_dir,prefix,pat)
    # example: /nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Raw/physionet.org/files/mimic-cxr/2.0.0/files/p10/p10000032/s50414267.txt
    txt_file_name = Path(os.path.join(pat_path,f'{study}.txt'))
    if txt_file_name.is_file():
        note = open(txt_file_name, "r", errors='ignore')
        img_note_text = note.read()
        note.close()
    else:
        return ''
    return img_note_text
    
## read jpg 
mimiciv_path = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Raw/physionet.org/files/'
## read corresponding notes
mimic_cxr_note_dir = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Raw/physionet.org/files/mimic-cxr/2.0.0/files/'

# read CXR metadata
try:
    df_mimic_cxr_metadata = dd.read_csv(mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', assume_missing=True, dtype={'dicom_id': 'object'}, blocksize=None)
except:
    df_mimic_cxr_metadata = pd.read_csv(mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv', dtype={'dicom_id': 'object'})
    df_mimic_cxr_metadata = dd.from_pandas(df_mimic_cxr_metadata, npartitions=7)
# only keep AP or PA 
df_mimic_cxr_metadata = df_mimic_cxr_metadata[df_mimic_cxr_metadata['ViewPosition'].isin(['PA','AP'])].compute()

df_mimic_cxr_metadata['note'] = df_mimic_cxr_metadata.apply(add_image_note,axis=1)

# read label
df_mimic_cxr_chexpert = dd.read_csv(mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv', assume_missing=True)
mimic_cxr_note_chexpert = dd.merge(df_mimic_cxr_metadata, df_mimic_cxr_chexpert, on=['subject_id','study_id'], how='inner')

# calculate the missing notes rate
note_missing_rate = sum(mimic_cxr_note_chexpert['note'] == '')/len(mimic_cxr_note_chexpert)
print('note_missing_rate:',note_missing_rate)


try:
    mimic_cxr_note_chexpert.compute().to_csv('mimic_cxr_note_chexpert.csv',index=False)
except Exception as e:
    print(e)

