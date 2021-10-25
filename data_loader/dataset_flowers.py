import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import typer
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

ROOT_DATA_DIR = "./data/"

logger = logging.getLogger(__name__)
app = typer.Typer()

NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]  # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/oxford_flowers102.py
IMAGE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"


def download_dataset(root_dir: str):
    root_dir = Path(root_dir)

    # check if files already exist
    images_exist = (root_dir / "102flowers.tgz").exists()
    labels_exist = (root_dir / "imagelabels.mat").exists()

    # download and unpack images
    if not images_exist:
        logger.info("downloading images...")
        torchvision.datasets.utils.download_and_extract_archive(IMAGE_URL, root_dir)

    # download labels
    if not labels_exist:
        logger.info("downloading labels...")
        torchvision.datasets.utils.download_url(LABELS_URL, root_dir)


@app.command()
def split_dataset(root_dir: str, target_dir: str, val_size=0.1, random_state=14, download=False):
    """
    Creates two CSVs with filepaths to the images in the original dataset.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if download:
        download_dataset(root_dir)

    root_dir = Path(root_dir)
    target_dir = Path(target_dir)
    labels_filename = root_dir / "imagelabels.mat"
    labels = loadmat(labels_filename)["labels"].flatten() - 1
    filepaths = [root_dir / "jpg" / f"image_{index+1:05}.jpg" for index in range(len(labels))]
    X_train, X_val, y_train, y_val = train_test_split(
        filepaths, labels, test_size=val_size, random_state=random_state, stratify=labels
    )

    train_dataset = {"filename": X_train, "label": y_train}
    val_dataset = {"filename": X_val, "label": y_val}
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving dataset splits to CSVs...")
    pd.DataFrame(train_dataset).to_csv(target_dir / "train_split.csv", index=False)
    pd.DataFrame(val_dataset).to_csv(target_dir / "val_split.csv", index=False)

    logger.info(f"Train split: {target_dir / 'train_split.csv'}")
    logger.info(f"Validation split: {target_dir / 'val_split.csv'}")


class OxfordFlowers102Dataset(Dataset):
    """
    Oxford 102 Category Flower Dataset
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

    Note: images are not shuffled by default.
    """

    def __init__(self, root_dir: str = ROOT_DATA_DIR, download: bool = False, transform=None):

        self.transform = transform
        self.root_dir = Path(root_dir)

        if download:
            download_dataset(self.root_dir)

        labels_filename = self.root_dir / "imagelabels.mat"
        # shift labels from 1-index to 0-index
        self.labels = loadmat(labels_filename)["labels"].flatten() - 1

    def __getitem__(self, index):
        filepath = self.root_dir / "jpg" / f"image_{index+1:05}.jpg"
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        return img, label, index

    def __len__(self):
        return len(self.labels)

    @property
    def classes(self):
        return NAMES


class OxfordFlowersDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=ROOT_DATA_DIR, batch_size=64, train_transforms=[]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = train_transforms
        self.train_sampler = None
        self.val_sampler = None

    def prepare_data(self):
        self.dataset = OxfordFlowers102Dataset(self.data_dir, transforms=self.transform)
        train_idx, valid_idx = self.get_sampler_indices()
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(valid_idx)

    def get_sampler_indices(self, val_size=0.1, shuffle=True, random_seed=14):
        num_train = len(self.dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        return train_idx, valid_idx

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=4)

    @property
    def len_train(self):
        return len(self.train_sampler)

    @property
    def len_valid(self):
        return len(self.val_sampler)