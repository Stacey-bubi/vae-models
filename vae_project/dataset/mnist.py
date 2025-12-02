from ..imports import *
from torchvision import datasets, transforms as tfm
from .transforms import Colorize

def get_dataset(
    root: str = "./data",
    train: bool = True,
    transforms: list = None,
):
    "Get a transformed MNIST dataset, optionally a colored one."
    transforms = (transforms or []).copy()
    transforms.append(tfm.ToTensor())
    return datasets.MNIST(root=root, train=train, download=True, transform=tfm.Compose(transforms))
