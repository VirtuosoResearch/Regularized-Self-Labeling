import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# a new transform
class MatchChannel(object):
    def __call__(self, pic):
        if pic.size()[0] == 1:
            assert len(pic.size()) == 3
            pic = pic.repeat(3,1,1)
        return pic

class BaseDataset(Dataset):

    def __init__(self, data) -> None:
        super(BaseDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        pass

class BaseDataLoader(DataLoader):
    # Data Augmentation defaults
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                MatchChannel(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            MatchChannel(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, valid_split, test_split, num_workers, collate_fn=default_collate):
        ''' Use validation split to check whether to generate validation dataset '''
        self.valid_split = valid_split
        self.test_split = test_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        if valid_split == 0.0:
            self.sampler = self._split_sampler_only_train()
            self.valid_sampler, self.test_sampler = None, None
        else:
            if test_split == 0.0:
                self.sampler, self.valid_sampler = self._split_sampler_valid(self.valid_split)
                self.test_sampler = None
            else:
                self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler_test(self.valid_split, self.test_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler_only_train(self):
        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)
        train_sampler = SubsetRandomSampler(idx_full)
        
        np.random.seed(0)
        self.shuffle = False

        return train_sampler

    def _split_sampler_valid(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def _split_sampler_test(self, valid_split, test_split):
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        # if isinstance(split, int):
        #     assert split > 0
        #     assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
        #     len_valid = split
        # else:
        assert valid_split < 1 and valid_split > 0
        assert test_split < 1 and test_split > 0
        assert valid_split + test_split < 1
        len_valid = int(self.n_samples * valid_split)
        len_test = int(self.n_samples * test_split) 

        valid_idx = idx_full[0:len_valid]
        test_idx = idx_full[len_valid:len_valid+len_test]
        train_idx = idx_full[len_valid+len_test:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = BaseDataLoader.test_transform
            return DataLoader(sampler=self.valid_sampler, **kwargs)

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            kwargs = deepcopy(self.init_kwargs)
            tmp_dataset = kwargs["dataset"]
            tmp_dataset.transform = BaseDataLoader.test_transform
            return DataLoader(sampler=self.test_sampler, **kwargs)
