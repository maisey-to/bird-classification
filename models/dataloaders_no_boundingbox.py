import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import decode_image

from BirdImageDataset import BirdImageDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset = BirdImageDataset(annotations_file='..\\archive\\CUB_200_2011\\images.txt', img_dir='..\\archive\\CUB_200_2011\\images')
# print(dataset.__getitem__(0))

# Loading the dataset
# Using the following reference for setting up alexnet
# https://www.digitalocean.com/community/tutorials/alexnet-pytorch

def get_train_valid_loader(data_dir, 
                           batch_size, 
                           augment, random_seed, 
                           validation_set_percentage=0.1, 
                           shuffle=True):
    # https://www.kaggle.com/code/givkashi/cub-image-classification-with-resnet34
    # """Normalizes images with Imagenet stats."""
    #imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    #return (im/255.0 - imagenet_stats[0])/imagenet_stats[1]

    # Define normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Define transforms
    valid_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
    ])

    labels_transform = torch.as_tensor
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.RandomCrop(227, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    else:
        train_transform = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                normalize,
            ])

    # Load the dataset
    train_dataset = BirdImageDataset(
        images_file='..\\archive\\CUB_200_2011\\images.txt',
        labels_file='..\\archive\\CUB_200_2011\\image_class_labels.txt',
        img_dir='..\\archive\\CUB_200_2011\\images',
        transform=train_transform,
        target_transform=labels_transform
    )

    valid_dataset = BirdImageDataset(
        images_file='..\\archive\\CUB_200_2011\\images.txt',
        labels_file='..\\archive\\CUB_200_2011\\image_class_labels.txt',
        img_dir='..\\archive\\CUB_200_2011\\images',
        transform=valid_transform,
        target_transform=labels_transform
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_set_percentage * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    labels_transform = torch.as_tensor

    dataset = BirdImageDataset(
        images_file='..\\archive\\CUB_200_2011\\images.txt',
        labels_file='..\\archive\\CUB_200_2011\\image_class_labels.txt',
        img_dir='..\\archive\\CUB_200_2011\\images',
        transform=transform,
        target_transform=labels_transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

# Bird Image Dataset
train_loader, valid_loader = get_train_valid_loader(None, 
                                                    batch_size=64, 
                                                    augment=False, 
                                                    random_seed=1)
test_loader = get_test_loader(None, 
                              batch_size=64)