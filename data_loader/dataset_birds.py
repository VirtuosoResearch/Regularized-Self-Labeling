""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict

class BirdDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels
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

        self.image_path = OrderedDict()
        self.labels_dict = {}
        self.labels = []

        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.image_id = []
        self.num_classes = 200

        # get image path from images.txt
        with open(os.path.join(self.data_dir, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                self.image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(self.data_dir, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                self.labels_dict[id] = int(label) - 1

        # get train/test image id from train_test_split.txt
        with open(os.path.join(self.data_dir, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                    self.labels.append(self.labels_dict[image_id])
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)
                    self.labels.append(self.labels_dict[image_id])

        # transform
        self.transform = transform

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(self.data_dir, 'images', self.image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        label = self.labels[item] # self.labels[image_id] 
        label = torch.tensor(label, dtype=torch.long)

        # return image and label
        return image, label, item  # count begin from zero

    def __len__(self):
        return len(self.image_id)


# if __name__ == '__main__':
#     ds = BirdDataset('train')
#     print(len(ds))
#     for i in range(0, 10):
#         image, label = ds[i]
#         print(image.shape, label)