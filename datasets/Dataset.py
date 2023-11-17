from pathlib import Path
import torchxrayvision.datasets as xrv_data
import numpy as np
import pandas as pd
from skimage.io import imread
import torch, random,os
from torch.utils.data import Dataset




def apply_transforms(sample, transform, seed=None, transform_seg=False):
    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

    return sample

class MIMIC_CXRDataset(xrv_data.Dataset):
    """
    Input:
        csvpath <python str>: path to the csv file
        views <python list>: list of views to use
        seed <python int>: seed for randomization
        unique_patients <python bool>: if True, only use one image per patient
        transforms <torchvision.transforms>: transforms to apply to the image
        xrv_normalize <python bool>: if True, normalize the image using the xrv normalize function
    """

    def __init__(self,
                 csvpath,
                 views=["PA", "AP"],
                 seed=42,
                 transforms=None,
                 xrv_normalize=0
                 ):

        super(MIMIC_CXRDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.img_root = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Data/Raw/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
        self.transforms = transforms
        self.xrv_normalize = xrv_normalize
        self.label_columns = ['Atelectasis',
                   'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                   'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding',
                   'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
                   'Support Devices']

        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        
        self.labels = self.csv[self.label_columns]
        self.labels = self.labels.fillna(0)
        self.labels = self.labels.replace(-1, 0)
        
        

    @property
    def csv_file(self):
        return self.csv
    
    def __len__(self):
        return self.csv.shape[0]
    
    def get_img_path(self,idx):
        row = self.csv.iloc[idx,:]
        prefix = 'p' + str(row['subject_id'])[:2]
        pat = 'p' + str(int(row['subject_id']))
        study = 's'+ str(int(row['study_id']))
        dicom_id = str((row['dicom_id'])) 
        pat_path = os.path.join(self.img_root,prefix,pat,study)
        self.jpg_name = Path(os.path.join(pat_path,f'{dicom_id}.jpg'))
        label = self.labels.iloc[idx,:]
        
        return self.jpg_name,label

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        
        img_path,label = self.get_img_path(idx)
        sample['label'] = list(label)
        sample['img_path'] = img_path.__str__()
        if self.xrv_normalize == 1:
            # single channel output renormliazed to [-1024, 1024]
            sample["img"] = xrv_data.normalize(imread(str(img_path)), maxval=255, reshape=True)
        elif self.xrv_normalize == 0:
            # single channel output renormliazed to [0, 1]
            sample["img"] = imread(str(img_path), as_gray=True).astype(np.float32) / 255.
        elif self.xrv_normalize == 2:
            # single channel output renormliazed to [0, 255]
            img = imread(str(img_path), as_gray=True)
            img = ((img - img.min()) / (img.max() - img.min())) * 255.
            img = img.astype('uint8')
            sample["img"] = img

        if self.transforms is not None:
            sample = apply_transforms(sample, self.transforms)

        return sample
    
    
if __name__ == '__main__':
    dataset = MIMIC_CXRDataset(csvpath = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/multimodal/mimic_cxr_note_chexpert.csv')
    print(len(dataset))
    sample = dataset[0]
    print(sample)