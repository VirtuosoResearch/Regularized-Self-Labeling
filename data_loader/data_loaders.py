from torchvision import datasets, transforms
from .base_data_loader import BaseDataLoader
from .dataset_caltech import Caltech256
from .dataset_flowers import OxfordFlowers102Dataset
from .dataset_aircrafts import AircraftDataset
from .dataset_birds import BirdDataset
from .dataset_cars import CarDataset
from .dataset_dogs import DogDataset
from .dataset_indoor import IndoorDataset

class MatchChannel(object):
    def __call__(self, pic):
        if pic.size()[0] == 1:
            assert len(pic.size()) == 3
            pic = pic.repeat(3,1,1)
        return pic

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, valid_split=0.0, num_workers=1, phase="train"):
        training = phase == "train"
        if training:
            trsfm = BaseDataLoader.train_transform
        else:
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class CaltechDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, idx_start, img_num, num_workers, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = Caltech256(
            self.data_dir, transform=trsfm, download=True, idx_start = idx_start, img_num=img_num
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=0, test_split=0, num_workers=num_workers)

class FlowerDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, valid_split, test_split, num_workers):
        trsfm = BaseDataLoader.train_transform
        self.data_dir = data_dir
        self.dataset = OxfordFlowers102Dataset(
            root_dir = self.data_dir, transform=trsfm, download=True 
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split, test_split, num_workers)

class AircraftsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = AircraftDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=0, test_split=0, num_workers=num_workers)

class BirdsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = BirdDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class CarsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = CarDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class DogsDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = DogDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)

class IndoorDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, num_workers, valid_split = 0, phase = "train"):
        if phase == "train":
            trsfm = BaseDataLoader.train_transform
        elif phase == "val" or phase == "test":
            trsfm = BaseDataLoader.test_transform
        self.data_dir = data_dir
        self.dataset = IndoorDataset(
            data_dir = self.data_dir, transform=trsfm, phase=phase
        )
        super().__init__(self.dataset, batch_size, shuffle, valid_split=valid_split, test_split=0, num_workers=num_workers)