
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class IndoorDataset(Dataset):

    def __init__(self, data_dir, transform, phase = "train"):
        self.data_dir = data_dir
        self.file_names = []
        if phase == "train":
            with open(os.path.join(self.data_dir, "TrainImages.txt")) as f:
                for line in f.readlines():
                    self.file_names.append(line.strip())
        elif phase == "test":
            with open(os.path.join(self.data_dir, "TestImages.txt")) as f:
                for line in f.readlines():
                    self.file_names.append(line.strip())

        # e.g. 'airport_inside/airport...0001.jpg' -> 'airport_inside'
        labels = [fn[:fn.find('/')] for fn in self.file_names]
        with open(os.path.join(self.data_dir, "name_to_id.json")) as f:
            label_to_ids = json.load(f)
        self.labels = [label_to_ids[label] for label in labels]

        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int):

        image = Image.open(os.path.join(self.data_dir, 'Images', self.file_names[index])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        return image, label, index