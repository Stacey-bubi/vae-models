from ..imports import *
from ..utils import *
from torchvision import datasets, transforms as tfm
from .transforms import Colorize, ToBatchTransform


def get_dataset(
    root: str = "./data",
    train: bool = True,
    transforms: list = None,
):
    "Get a transformed MNIST dataset, optionally a colored one."
    transforms = (transforms or []).copy()
    transforms.append(tfm.ToTensor())
    return datasets.MNIST(root=root, train=train, download=True, transform=tfm.Compose(transforms))


class ColoredMNIST(datasets.MNIST):
    "An Colored MNIST dataset to match `torch_fidelity` api (slow)"

    def __init__(
        self,
        root,
        train=True,
        download=False,
        color_type: Literal["random", "label", "random_discrete"] = "label",
    ):
        pre_tfms = tfm.Compose([tfm.ToTensor(), tv.transforms.Pad(2)])
        super().__init__(root, train, pre_tfms, download=download)
        self.tfms = [Colorize(color_type), ToBatchTransform(to_uint8_img)]
        self.n_classes = len(self.classes)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img, label = torch.utils.data.default_collate([item])
        for tfm in self.tfms:
            img, label = tfm((img, label))
        return img.squeeze(0)
