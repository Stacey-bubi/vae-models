import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .imports import *


def normalize_img(x: t.Tensor):
    '''Normalizes image to 0-1 range'''
    if x.min() >= 0 and x.max() <=1:
        return x
    return (x-x.min())/(x.max()-x.min())

def show_img(x: t.Tensor, ax=None, title=''):
    """Displays a single image tensor of shape (C, H, W)."""
    if ax is None:
        ax = plt.subplot()
    ax.imshow(x.cpu().detach().permute(1, 2, 0))
    ax.set_title(title)
    return ax


def show_imgs(xb: t.Tensor, ax=None, title=''):
    """Arranges and displays a batch of image tensors (B, C, H, W) in a grid."""
    nrow = int(xb.shape[0] ** 0.5)
    xb = make_grid(xb, nrow=nrow, pad_value=1)
    return show_img(xb, ax, title)


# Selects CUDA if available, otherwise defaults to CPU.
default_device = t.device("cuda" if t.cuda.is_available() else "cpu")


def to_device(x, device=default_device):
    """Recursively moves tensors or collections (lists, tuples, dicts) of tensors to a device."""
    if isinstance(x, t.Tensor):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return [to_device(el, device) for el in x]
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    return x


def random_seed(n=42):
    """Set random seed for reproducibility"""
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)


def noop(x):
    '''returns input'''
    return x
