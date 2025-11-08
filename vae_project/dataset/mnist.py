# %%
from typing import Callable
from functools import partial
import torch as t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as tfm


def colorize(img: t.Tensor, label: t.Tensor):
    "Colorizes a single-channel image with a random color."
    c = t.rand(3, 1, 1)  # random RGB color
    return img * c


def get_dataset(
    root: str = "./data",
    train: bool = True,
    transforms: list = None,
    colorize_fn: Callable = None,
):
    "Get a transformed MNIST dataset, optionally a colored one."
    transforms = transforms or []
    transforms.append(tfm.ToTensor())
    cls = (
        datasets.MNIST
        if colorize_fn is None
        else partial(ColoredMNIST, colorize_fn=colorize_fn)
    )
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

    def __init__(self, root, colorize_fn, train=True, transform=None, download=False):
        super().__init__(root, train, transform, download=download)
        self.colorize_fn = colorize_fn

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.colorize_fn(img, label)
        return img, label
# %%

ds = get_dataset("../data", colorize_fn=colorize)
# %%

dl = DataLoader(ds, 4)
xb, yb = next(iter(dl))
