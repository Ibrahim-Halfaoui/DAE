import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import sys
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import glob
from torch.utils import data
import os
import random
dir_ = os.getcwd()


res_dim = (128, 256)
class DataloaderDocs(data.Dataset):
    def __init__(self, datadir):
        super(DataloaderDocs, self).__init__()
        if not os.path.isdir(datadir):
            print('cannot find dataset')
        else:
            os.chdir(datadir)
            self.img_files = [datadir + '/' + f for f in glob.glob("*.png")]
            print(len(self.img_files), ' images are available in this folder')

    def augment(self, image):
        self.transforms_photo = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomInvert(p=0.2),
            # transforms.RandomPosterize(bits=2, p=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.Grayscale(),
            # transforms.Pad(padding=(random.randint(0,25)),fill=0, padding_mode='constant'),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.09, 0.5)),
            transforms.RandomSolarize(threshold= 10, p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.5),
            transforms.RandomAutocontrast(p=0.2),
            # transforms.RandomEqualize(p=0.5),

        ])


        # self.transforms_geo = transforms.Compose([
        #     transforms.RandomVerticalFlip(0.1),
        #     transforms.RandomAffine(degrees=(5, 70), translate=(0.01, 0.1), scale=(0.5, 0.75)),
        #     transforms.RandomPerspective(distortion_scale=0.5, p=0.6),
        #     transforms.RandomRotation(degrees=(0, 180)),
        #     transforms.RandomErasing(p=0.05, scale=(0.02, 0.13), ratio=(0.3, 1.3), value=0, inplace=False),
        # ])
        self.transforms_geo = [None] * 5
        self.transforms_geo[0] = transforms.RandomVerticalFlip(0.4)
        self.transforms_geo[1] = transforms.RandomAffine(degrees=(5, 45), translate=(0.01, 0.1), scale=(0.5, 0.75))
        self.transforms_geo[2] = transforms.RandomPerspective(distortion_scale=0.3, p=1)
        self.transforms_geo[3] = transforms.RandomRotation(degrees=(-15, 15))
        self.transforms_geo[4] = transforms.RandomErasing(p=0.05, scale=(0.02, 0.13), ratio=(0.3, 1.3), value=0, inplace=False)


        # photo aug
        image = self.transforms_photo(image)
        image = transforms.functional.adjust_sharpness(image, sharpness_factor=0.8)
        # if np.random.random() > 0.5:
        #     image = self.transforms_geo(image)
        if np.random.random() < 0.1:
            image = self.transforms_geo[0](image)
        elif np.random.random() > 0.1 and np.random.random() < 0.3:
            image = self.transforms_geo[1](image)
        elif np.random.random() > 0.3 and np.random.random() < 0.7:
            image = self.transforms_geo[2](image)
        elif np.random.random() > 0.7 and np.random.random() < 0.8:
            image = self.transforms_geo[3](image)
        else:
            image = self.transforms_geo[4](image)

        return image

    def prep_or(self, image):
        self.trafo = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),

        ])
        im_t = self.trafo(image)
        return im_t

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path)
        data = data.astype(np.float32)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data, res_dim)
        data /= 255.
        # data = Image.open(img_path).convert('RGB')
        # data = data.resize(res_dim, Image.NEAREST)
        # threshold = 125
        # data = data.point(lambda p: p > threshold and 255)

        data_or = self.prep_or(data)
        data_aug = self.augment(data)
        # data = data + (0.1 ** 0.5) * torch.randn(data.size())
        return data_aug, data_or

    def __len__(self):
        return len(self.img_files)