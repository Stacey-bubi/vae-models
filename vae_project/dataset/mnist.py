from typing import Callable
from functools import partial
import torch as t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as tfm


def get_dataset(
    root: str = "./data",
    train: bool = True,
    transforms: list = None,
    colored: bool = True,
):
    "Get a transformed MNIST dataset, optionally a colored one."
    transforms = transforms or []
    transforms.append(tfm.ToTensor())
    cls = datasets.MNIST if not colored else ColoredMNIST
    return cls(root=root, train=train, download=True, transform=tfm.Compose(transforms))


def to_dataloaders(train_dataset, val_dataset=None, batch_size=32, val_batch_size=None):
    if not val_batch_size:
        val_batch_size = batch_size
    return (
        DataLoader(ds, batch_size, shuffle=True)
        for ds in zip([train_dataset, val_dataset], [batch_size, val_batch_size])
        if ds
    )


class ColoredMNIST(datasets.MNIST):
    "An MNIST dataset that applies a `colorize_fn` to each image."

    def __init__(self, root, train=True, transform=None, download=False, random_colors=False):
        super().__init__(root, train, transform, download=download)
        self.colorize_fn = self.colorize_rand if random_colors else self.colorize_cls
        self.n_classes = len(self.classes)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.colorize_fn(img, label)
        return img, label


    def colorize_rand(self, img: t.Tensor, label: t.Tensor):
        "Colorizes a single-channel image with a random color."
        c = t.rand(img.shape[0], 3, 1, 1)  # random RGB color
        return img * c


    def colorize_cls(self, img: t.Tensor, label: t.Tensor):
        "Colorizes a single-channel image based on class label."
        h = t.as_tensor(label, device=img.device).float()/self.n_classes*6
        c = t.stack([(h-3).abs()-1, 2-(h-2).abs(), 2-(h-4).abs()], -1).clamp(0,1)
        return img * c[...,None,None]