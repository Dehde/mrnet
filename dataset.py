import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine


class MRDataset(data.Dataset):
    def __init__(self, root_dir, task, plane, train=True, transform=None, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            csv_path = self.root_dir + 'train-{0}.csv'.format(task)
            print(csv_path)
            self.records = pd.read_csv(csv_path, header=None, names=['id', 'label'])
        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])
        # self.records = self.records[:100]

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = torch.FloatTensor(self.labels[index])
        if label == 1:
            label = torch.FloatTensor([0, 1])
        elif label == 0:
            label = torch.FloatTensor([1, 0])

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #      weight = np.array([self.weights[1]])
        #      weight = torch.FloatTensor(weight)
        # else:
        #      weight = np.array([self.weights[0]])
        #      weight = torch.FloatTensor(weight)

        return array, label


class MRDatasetMerged(data.Dataset):
    def __init__(self, root_dir, train=True, transform=None, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.planes = ["sagittal", "coronal", "axial"]
        if self.train:
            csv_path = self.root_dir + 'train_labels.csv'
            self.records = pd.read_csv(csv_path, header=1, names=['id',"abnormal","acl","meniscus"])
            self.folder_path = self.root_dir + 'train/{plane}/'
        else:
            transform = None
            csv_path = self.root_dir + 'valid_labels.csv'
            self.records = pd.read_csv(csv_path, header=1, names=['id', "abnormal", "acl", "meniscus"])
            self.folder_path = self.root_dir + 'valid/{plane}/'
        # self.records = self.records[:100]

        self.paths = [self.folder_path + str(filename).zfill(4) +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records.loc[:, ["abnormal", "acl", "meniscus"]]
        self.labels.loc[:, ["abnormal", "acl", "meniscus"]] = self.labels.loc[:, ["abnormal", "acl", "meniscus"]].astype(int)
        self.labels = self.labels.values

        self.transform = transform
        # TODO: re-add weights functionality
        # if weights is None:
        #     pos = np.sum(self.labels)
        #     neg = len(self.labels) - pos
        #     self.weights = torch.FloatTensor([1, neg / pos])
        # else:
        #     self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        mri_series = self.load_merged_arr(index)
        label = torch.FloatTensor(self.labels[index])

        if self.transform:
            array = self.transform(mri_series)
        else:
            # Maybe: turn grayscale image to 3 channels for use of pretrained models
            array = np.stack((mri_series,)*3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #      weight = np.array([self.weights[1]])
        #      weight = torch.FloatTensor(weight)
        # else:
        #      weight = np.array([self.weights[0]])
        #      weight = torch.FloatTensor(weight)
        return array, label

    def load_merged_arr(self, idx):
        curr_path_template = self.paths[idx]
        merged_arrs = np.vstack([np.load(curr_path_template.format(plane=plane)) for plane in self.planes])
        return merged_arrs

if __name__ == "__main__":
    ds = MRDatasetMerged('./data/')
    for b in ds:
        print(b, b[0].shape, b[1].shape)
        break
