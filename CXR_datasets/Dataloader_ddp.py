import numpy as np
import torch.utils.data as module_data
import torch.utils.data.dataloader as module_default_dataloader
from torchvision import transforms
import CXR_datasets.Dataset as module_dataset
import utils.ddp_samplers as module_ddp_sampler

class DistributedSplitDataLoader(module_data.DataLoader):

    """
    Loading data from a dataset with a split ratio for validation set.
    The split can only be done randomly.
    
    Input:
        dataset <python str>: the name of the dataset class
        batch_size <python int>: the batch size
        shuffle <python bool>: whether to shuffle the data
        validation_split <python float or int>: the ratio of the validation set
        patient_split <python str>: the name of class attribute that correspond to  
            the csv file and the patient ID column name. The format is "csv;PatientID"
        num_workers <python int>: the number of workers for data loader

    """
    def __init__(self, dataset, batch_size, validation_split, num_workers=8, 
                 collate_fn=module_default_dataloader.default_collate,
                 train_sampler_class=module_ddp_sampler.CustomizedDistributedSampler, 
                 valid_sampler_class=module_ddp_sampler.CustomizedDistributedEvalSampler,
                 rank=None, num_replicas=None):
        
        self.dataset = dataset
        self.validation_split = validation_split
        self.n_samples = len(dataset)

        if self.validation_split > 0: # the validation split is done randomly
            self.train_indice, self.valid_indice = self._split_indices(
                self.validation_split)
        else:
            self.train_indice, self.valid_indice = np.arange(self.n_samples), None

        self._sampler = train_sampler_class(
            self.train_indice, rank=rank, 
            num_replicas=num_replicas)
        
        self._valid_sampler = valid_sampler_class(
            self.valid_indice, rank=rank, 
            num_replicas=num_replicas) if self.valid_indice is not None else None
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'drop_last': True, 
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory':True
        }
        
        super().__init__(sampler=self._sampler, **self.init_kwargs)
        
    def _split_indices(self, split):
        """
        Split the dataset indices randomly into training and validation.
        The self.n_samples is updated to the number of training samples.
        """

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured "+ \
                                            "to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        self.n_samples = len(train_idx)

        return train_idx, valid_idx
    
    def get_val_loader(self):
        if self._valid_sampler is None:
            return None
        else:
            return module_data.DataLoader(sampler=self._valid_sampler, **self.init_kwargs)


class Dist_MIMIC_CXRDataLoader(DistributedSplitDataLoader):
    """
    Chest X-ray data loading base on the DistributedSplitDataLoader

    Input:
        image_size <python int>: the shape of the resized image
        rank <python int>: the rank of the current process
        num_replicas <python int>: the number of processes, equals to the world size

    - The images are normalized to [0, 1] in dataset, 
    - In this dataloader, images are resized to (image_size, image_size), 
        turned to RGB.
    - The dimession of the output tensor is (batch_size, 3, image_size, image_size)
    """
    def __init__(self, data_dir, batch_size, image_size=256, validation_split=0.15, 
                 num_workers=8, rank=None, num_replicas=None):

        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Turn grayscale to RGB
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        MIMIX_data = module_dataset.MIMIC_CXRDataset(data_dir, transforms=trsfm, xrv_normalize=2)
        # COVID_QU_EX_data = module_dataset.COVID_QU_EX(csvpath=data_apth_covid, transforms=trsfm, xrv_normalize=2)
        self.dataset = MIMIX_data

        super().__init__(self.dataset, batch_size, validation_split, 
                         num_workers=num_workers, rank=rank, num_replicas=num_replicas)