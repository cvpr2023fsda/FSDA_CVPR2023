import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np

class STL10Dataset(Dataset) :
    def __init__(self, add_noise='gaussian_noise', transform=None, severity=1):
        super(STL10Dataset, self).__init__()

        self.transform = transform
        self.severity = severity
        self.data_path = 'dataset/STL-10-C'
        self.npy_data = np.load(os.path.join(self.data_path, '{}.npy'.format(add_noise)))[self.severity * 8000 : (self.severity + 1) * 8000]
        self.target = np.load(os.path.join(self.data_path, 'target.npy'))


    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        image = self.npy_data[idx]
        label = self.target[idx]

        if self.transform :
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label

class CIFAR100Dataset(Dataset) :
    def __init__(self, add_noise='gaussian_noise', transform=None, severity=1):
        super(CIFAR100Dataset, self).__init__()

        self.transform = transform
        self.severity = severity
        self.data_path = 'dataset/CIFAR-100-C'
        self.npy_data = np.load(os.path.join(self.data_path, '{}.npy'.format(add_noise)))[self.severity * 10000 : (self.severity + 1) * 10000]
        self.target = np.load(os.path.join(self.data_path, 'target.npy'))


    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        image = self.npy_data[idx]
        label = self.target[idx]

        if self.transform :
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label

class CIFAR10Dataset(Dataset) :
    def __init__(self, add_noise='gaussian_noise', transform=None, severity=1):
        super(CIFAR10Dataset, self).__init__()

        self.transform = transform
        self.severity = severity
        self.data_path = 'dataset/CIFAR-10-C'
        self.npy_data = np.load(os.path.join(self.data_path, '{}.npy'.format(add_noise)))[self.severity * 10000 : (self.severity + 1) * 10000]
        self.target = np.load(os.path.join(self.data_path, 'target.npy'))

    def __len__(self):
        return len(self.npy_data)

    def __getitem__(self, idx):
        image = self.npy_data[idx]
        label = self.target[idx]

        if self.transform :
            image = transforms.ToPILImage()(image)
            image = self.transform(image)

        return image, label