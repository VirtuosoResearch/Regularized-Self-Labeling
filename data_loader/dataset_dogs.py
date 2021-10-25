""" Stanford Dogs (Dog) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

class DogDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Dogs images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, data_dir, transform, phase='train'):
        self.data_dir = data_dir

        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.num_classes = 120

        if phase == 'train':
            list_path = os.path.join(self.data_dir, 'train_list.mat')
        else:
            list_path = os.path.join(self.data_dir, 'test_list.mat')

        list_mat = loadmat(list_path)
        self.images = [f.item().item() for f in list_mat['file_list']]
        self.labels = [f.item() for f in list_mat['labels']]
        self.labels = [label - 1 for label in self.labels]
        # transform
        self.transform = transform

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(self.data_dir, 'Images', self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        label = self.labels[item] 
        label = torch.tensor(label, dtype=torch.long)
        # return image and label
        return image, label, item  # count begin from zero

    def __len__(self):
        return len(self.images)


# if __name__ == '__main__':
#     ds = DogDataset('train')
#     # print(len(ds))
#     for i in range(0, 1000):
#         image, label = ds[i]
#         # print(image.shape, label)