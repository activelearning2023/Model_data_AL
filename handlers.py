import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import TensorDataset,DataLoader,Dataset
from monai import transforms
from seed import setup_seed
import torch
import albumentations as A


setup_seed(42)
# get dataloader
# online data augumentation
keys = ("image", "label")
class aug():
    def __init__(self):
        self.random_rotated = transforms.Compose([
            transforms.AddChanneld(keys),  # 增加通道，monai所有Transforms方法默认的输入格式都是[C, W, H, ...],第一维一定是通道维
            transforms.RandRotate90d(keys, prob=1, max_k=3, spatial_axes=(0, 1), allow_missing_keys=False),
            transforms.RandFlipd(keys, prob=1, spatial_axis=(0, 1), allow_missing_keys=False),
            transforms.RandGaussianNoised(keys, prob=0.1, mean=0.0, std=0.1, allow_missing_keys=False),
#             transforms.NormalizeIntensityd(keys, allow_missing_keys=False),
            transforms.ToTensord(keys)
        ])
    
    def forward(self,x):
        x = self.random_rotated(x)
        return x
    

class MSSEG_Handler_2d(Dataset):
    def __init__(self,image,label,mode="train"):
        self.image=np.array(image)
        self.label=np.array(label)
        if mode=="train":
            self.transform = A.Compose([
                A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, always_apply=False, p=0.5),
                A.Flip(p=0.5),
                A.Rotate (limit=90, interpolation=1,always_apply=False, p=0.5),
#                 A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
#                 A.Resize(336,336, interpolation=3, always_apply=True, p=1),
                # A.ToTensor(),
        ]) 
        else:
            self.transform=None
            
    def __len__(self):
        return len(self.label)
    def __getitem__(self,index): 
        img = self.image[index].astype(np.float32)
        label = self.label[index].astype(np.uint8)
        if self.transform!=None: 
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        img = torch.tensor(img)
        label = torch.tensor(label)
        return img, label, index



