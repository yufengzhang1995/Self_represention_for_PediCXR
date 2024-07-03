import kornia as module_kornia
import torch.nn as nn
import torch

class BestCombineAugmentation(nn.Module):
    """
    Module to perform data augmentation using Kornia on torch tensors.

    This augmentation comes from the paper "Exploring Image Augmentations for Siamese 
        Representation Learning with Chest X-Rays",in which the authors report that  
        combining random resize crop with distortion resulted in the best augmentation pair.
    
    The details of the augmentation are as follows:
        1. Brightness/contrast adjustment is implemented using ColorJitter with  
            brightness and contrast arguments (λ) set to 0.5.
        2. RandomResizedCrop to construct crops with random scale of 0.2 − 1.0  
            for pair-wise evaluation and 0.3 − 0.9 for t_θ, and the default 
            parameters for aspect ratio (3/ 4 − 4/ 3).
    """
    def __init__(self, image_size: int = 224, imagenet_norm=True) -> None:
        super().__init__()

        self.transforms = nn.Sequential(     
            module_kornia.augmentation.RandomResizedCrop(size=(image_size, image_size), 
                                         scale=(0.3, 0.9), ratio=(3.0 / 4.0, 4.0 / 3.0), p=1.0),
            module_kornia.augmentation.RandomBrightness(brightness=(0.5, 0.5), p=0.7),
            module_kornia.augmentation.RandomContrast(contrast=(0.5, 0.5), p=0.7),
        )

        if imagenet_norm:
            self.normalize = module_kornia.augmentation.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
        else:
            self.normalize = nn.Identity()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.transforms(x)) # BxCxHxW    

class BasicAugmentation(nn.Module):
    """
    Module to perform data augmentation using Kornia on torch tensors.
    """
    def __init__(self, image_size: int = 224, imagenet_norm=True) -> None:
        super().__init__()
        self.transforms = nn.Sequential(     
            module_kornia.augmentation.RandomRotation(degrees=15.0, p=0.5),
            module_kornia.augmentation.RandomResizedCrop(size=(image_size, image_size), 
                                         scale=(0.7, 0.95), ratio=(4.0 / 5.0, 5.0 / 4.0), p=0.8),
        )
        if imagenet_norm:
            self.normalize = module_kornia.augmentation.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
        else:
            self.normalize = nn.Identity()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.transforms(x)) # BxCxHxW


class ImageNetAugmentation(nn.Module):
    """
    Module to perform imagenet augmentation using Kornia on torch tensors.
        The image_size argument is not used, just take palce for compatibility.
    """
    def __init__(self, image_size: int = 224) -> None:
        super().__init__()

        self.normalize = module_kornia.augmentation.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]))

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) # BxCxHxW
    
class DINOAugmentation(nn.Module):
    def __init__(self,
                 global_crops_scale: tuple = (0.8, 0.99),
                 local_crops_scale: tuple = (0.4, 0.7),
                 local_crops_number: int = 4,
                 image_size: int = 256,
                 imagenet_norm: bool = True):
        super().__init__()
        if imagenet_norm:
            self.normalize = module_kornia.augmentation.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]))
        else:
            self.normalize = nn.Identity()

        self.flip_and_color_jitter = nn.Sequential(     
            module_kornia.augmentation.RandomHorizontalFlip(p=0.5),
            module_kornia.augmentation.RandomBrightness(brightness=(0.5, 0.5), p=0.7),
            module_kornia.augmentation.RandomContrast(contrast=(0.5, 0.5), p=0.7),
        )

        self.global_trsf1 = nn.Sequential(
            module_kornia.augmentation.CenterCrop(size=image_size),
            self.normalize
        )

        self.global_trsf2 = nn.Sequential(     
            # random rotation
            module_kornia.augmentation.RandomRotation(degrees=15.0, p=0.5),
            # random resized crop
            module_kornia.augmentation.RandomResizedCrop(size=(image_size, image_size), 
                                        scale=global_crops_scale, ratio=(4/5, 5/4), p=1.0),
            # random horizontal flip and color jitter
            self.flip_and_color_jitter,
            # gaussian blur
            module_kornia.augmentation.RandomGaussianBlur(kernel_size=(1, 1), 
                                                          sigma=(0.1, 1.0), p=0.5),
            self.normalize
        )

        self.local_crops_number = local_crops_number
        self.local_trsf =  nn.Sequential(
            module_kornia.augmentation.RandomRotation(degrees=15.0, p=0.5),
            module_kornia.augmentation.CenterCrop(size=4*image_size//5, p=1.0),
            module_kornia.augmentation.RandomResizedCrop(size=(image_size//2, image_size//2),
                                        scale=local_crops_scale, ratio=(4/5, 5/4), p=1.0),
            self.flip_and_color_jitter,
            self.normalize
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, image: torch.Tensor):
        crops = []
        crops.append(self.global_trsf1(image))
        crops.append(self.global_trsf2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_trsf(image))
        return crops

