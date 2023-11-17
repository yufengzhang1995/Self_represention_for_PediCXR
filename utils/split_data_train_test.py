
import datetime as dt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os
from pathlib import Path

tbl = pd.read_csv('mimic_cxr_note_chexpert.csv')
label_columns = ['Atelectasis',
                   'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                   'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding',
                   'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
                   'Support Devices']

ss = StratifiedShuffleSplit(n_splits=2, test_size=0.20, random_state=42)
labels = tbl[label_columns]
labels = labels.fillna(0)
labels = labels.replace(-1, 0)
tbl['level'] = np.argmax(labels.to_numpy(), axis=1)
index = list(ss.split(tbl.iloc[:,:-1], tbl.iloc[:,-1]))[0]
train_tbl = tbl.iloc[index[0],:]
test_tbl = tbl.iloc[index[1],:]

train_tbl.to_csv('train_cxr_note.csv',index=False)
test_tbl.to_csv('test_cxr_note.csv',index=False)