import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import numpy  as np
import cv2
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256),cv2.INTER_NEAREST),
    transforms.ToTensor(),
])

transform1 = transforms.Compose([
    transforms.Resize((256, 256),cv2.INTER_NEAREST),
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'seg'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        image_name=segment_name.replace("reference","oct")
        segment_path = os.path.join(self.path, 'seg', segment_name)
        image_path = os.path.join(self.path, 'image', image_name)
        segment_image = Image.open(segment_path)
        image =Image.open(image_path)
        return transform(image), torch.Tensor(np.array(transform1(segment_image))).long()







