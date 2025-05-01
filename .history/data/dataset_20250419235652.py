# Custom PyTorch Dataset class that loads images and boxes

import torch
from torch.utils.data import Dataset
from PIL import image
import os

class WiderFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None)
