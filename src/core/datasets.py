import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from typing import Any, Dict, List

from .aux import image_read

resample_modes = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}


class DDFMDataset(Dataset):
    def __init__(self, dir: Dict[int, List[str]], mode: str = 'L'):
        self.images_1, self.images_2 = dir[0], dir[1]
        self.mode = mode
    
    def __len__(self):
        return len(self.images_1)
    
    def __getitem__(self, idx):
        img_1 = image_read(self.images_1[idx], self.mode)/255.0
        img_2 = image_read(self.images_2[idx], self.mode)/255.0
        
        img_1 = img_1 * 2 - 1
        img_2 = img_2 * 2 - 1

        # Crop to make divisible
        scale = 32
        h, w = img_1.shape[:2]
        h = h - h % scale
        w = w - w % scale

        img_1 = ((torch.FloatTensor(img_1))[:, :h, :w]).permute(2,0,1)
        img_2 = ((torch.FloatTensor(img_2))[:, :h, :w]).permute(2,0,1)
        assert img_1.shape == img_2.shape, 'Image shape mismatch.'

        return {'M_1': img_1, 'M_2': img_2, 'Index': idx}

class SRDataset(Dataset):
    def __init__(self, dir: str, scale_factor: int, resample: str = 'bicubic', mode: str = 'L', **kwargs: Dict[str, Any]):
        self.lr_dir = dir
        self.scale_factor = scale_factor
        
        if resample in resample_modes.keys():
            self.resample_map = resample_modes[resample]
        else:
            raise ValueError(f'resample mode {resample} not supported')
        
        self.image_list = [os.path.join(self.lr_dir,f) for f in os.listdir(self.lr_dir) if f.endswith('.png')]

        self.mode = mode
        
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        lr_img = Image.open(self.image_list[idx]).convert(self.mode)
        h, w = lr_img.size
        sr_img = lr_img.resize((int(w * self.scale_factor), int(h * self.scale_factor)), self.resample_map)
        sr_img = 2. * F.to_tensor(sr_img) - 1

        return {'SR': sr_img, 'Index': idx, 'Name': self.image_list[idx].split('/')[-1]}