import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image


# Creating Custom Dataset with Bird Images 
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class BirdImageDataset(Dataset):
    def __init__(self, images_file, labels_file, img_dir, transform=None, target_transform=None):
        self.images = pd.read_csv(images_file, header=None)
        self.labels = pd.read_csv(labels_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_id, file_name = self.images.iloc[idx, 0].split(" ")
        image_id = int(image_id)
        img_path = os.path.join(self.img_dir, file_name)
        image = decode_image(img_path)
        image_float = image.to(torch.float32) / 255.0
        label = int(self.labels.iloc[image_id - 1, 0].split(" ")[1])
        if self.transform:
            image_float = self.transform(image_float)
        if self.target_transform:
            label = self.target_transform(label)
        return image_float, label

dataset = BirdImageDataset(images_file='..\\archive\\CUB_200_2011\\images.txt', labels_file='..\\archive\\CUB_200_2011\\image_class_labels.txt', img_dir='..\\archive\\CUB_200_2011\\images')
dataset.__getitem__(65)
