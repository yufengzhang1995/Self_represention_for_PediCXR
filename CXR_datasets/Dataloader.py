from pathlib import Path
import numpy as np
import torch.utils.data as module_data
import torch.utils.data.dataloader as module_default_dataloader
import torch.utils.data.sampler as module_sampler
from torchvision import transforms
import torchxrayvision.datasets as module_xrv_data
import torch
from CXR_datasets.Dataset import MIMIC_CXRDataset

def apply_transforms(sample, transform, seed=None, transform_seg=False):
    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

    return sample


class SplitDataLoader(module_data.DataLoader):

    def __init__(self, dataset, batch_size, shuffle=True,
                 split_method="patient-wise", split_ratio=0.3,num_workers=0,seed = 42, 
                 collate_fn=module_default_dataloader.default_collate):
        
        self.dataset = dataset
        self.table = self.dataset.csv
        self.shuffle = shuffle
        self.split_method = split_method
        self.split_ratio = split_ratio
        self.seed = seed
        self.n_samples = len(dataset)
        self._split_sampler()

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        
        super().__init__(**self.init_kwargs)

    def _split_sampler(self):

        if self.split_method == 'patient-wise':
            patients = self.table['subject_id'].unique()
        else:
            patients = np.arange(self.table['subject_id'].shape[0])
            
        patient_col = 'subject_id'

        np.random.seed(self.seed)
        np.random.shuffle(patients)

 
        len_valid = int(len(patients) * self.split_ratio)

        valid_patients = patients[0:len_valid]
        train_patients = np.delete(patients, np.arange(0, len_valid))

        self.train_idx = list(self.table[self.table[patient_col].isin(train_patients)].index)
        self.valid_idx = list(self.table[self.table[patient_col].isin(valid_patients)].index)

        self.train_sampler = module_sampler.SubsetRandomSampler(self.train_idx)
        self.valid_sampler = module_sampler.SubsetRandomSampler(self.valid_idx)
        self.n_samples = len(train_patients)
        
        self.shuffle = False


    def get_data_loader(self):
        train_loader = module_data.DataLoader(sampler=self.train_sampler, **self.init_kwargs)
        val_loader = module_data.DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        return train_loader,val_loader
        


class MIMIC_CXRDataLoader(SplitDataLoader):

    def __init__(self, data_dir, batch_size, image_size,  
                 split_method="patient-wise", split_ratio=0.3, num_workers=0, xrv_normalize=False):

        if xrv_normalize:
            # input channel = 1
            trsfm = transforms.Compose([
                module_xrv_data.XRayCenterCrop(), # just to make it square
                module_xrv_data.XRayResizer(image_size)
            ])
        else:
            # input channel = 3
            trsfm = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
        self.dataset = MIMIC_CXRDataset(data_dir,transforms=trsfm, xrv_normalize=xrv_normalize)

        super().__init__(self.dataset, batch_size,
                         num_workers=num_workers, split_method=split_method,split_ratio = split_ratio)
        
    
    def get_dataset(self):
        return self.dataset
    
    
if __name__ == '__main__': 
    
    loader = MIMIC_CXRDataLoader(data_dir = '/nfs/turbo/med-kayvan-lab/Projects/HeartFailure/Code/Yufeng/multimodal/mimic_cxr_note_chexpert.csv'
                           ,batch_size = 4, image_size = 224,
                           )
    train_loader,val_loader = loader.get_data_loader() 
    num_batches = len(train_loader)
    print("Number of batches in train_loader:", num_batches)
    batch_size = train_loader.batch_size
    print("Batch size of train_loader:", batch_size)

    num_batches = len(val_loader)
    print("Number of batches in val_loader:", num_batches)
    batch_size = val_loader.batch_size
    print("Batch size of val_loader:", batch_size)
    
    for sample in train_loader:
        print(sample)
        break 