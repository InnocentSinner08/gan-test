from io import BytesIO

import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset wrapper for StyleGAN2 training.

    Auto-downloads CIFAR-10 via torchvision. Returns only images (no labels)
    to match the MultiResolutionDataset interface.
    """

    def __init__(self, root='./data', size=32, train=True, data_ratio=1.0):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=self.transform,
        )

        # Support using a subset of the data (e.g., data_ratio=0.1 for 10%)
        total = len(self.dataset)
        self.num_samples = int(total * data_ratio)
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, _label = self.dataset[self.indices[index]]
        return img
