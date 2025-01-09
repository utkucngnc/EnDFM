import cv2
import logging
import numpy as np
import os
from typing import List, Dict

logger = logging.getLogger('base')

image_modes = ['RGB', 'YCrCb', 'L']

def check_integrity(dirs: List[str]) -> Dict[str, List[str] | str]:
    assert len(dirs) == 2, "Expected 2 directories, found {len(dirs)}."
    for dir in dirs:
        if not os.path.exists(dir):
            raise FileNotFoundError(f"Directory {dir} not found.")
        if not os.path.isdir(dir):
            raise NotADirectoryError(f"{dir} is not a directory.")
        if not os.listdir(dir):
            raise ValueError(f"{dir} is empty.")
    path_1, path_2 = dirs
    singletons = [img for img in set(os.listdir(path_1)) ^ set(os.listdir(path_2)) if img.endswith('.png')]
    images = [img for img in set(os.listdir(path_1)).intersection(os.listdir(path_2)) if img.endswith('.png')]
    integrity_score = f'{len(images)}/{len(images) + len(singletons)}'

    return {'Singletons': singletons, 'Images': images, 'Integrity Score': integrity_score}


def image_read(path: str, mode: str = 'RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in image_modes, 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'L':  
        img = np.expand_dims(np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)), axis=-1)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def check_mode(mode: str):
    if mode.upper() in image_modes:
        mode = mode.upper()
        return mode
    else:
        raise ValueError(f'image mode {mode} not supported')