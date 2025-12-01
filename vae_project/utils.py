import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch as t


def show_img(x: t.Tensor, ax=None):
    """Displays a single image tensor of shape (C, H, W)."""
    if ax is None: ax = plt.subplot()
    ax.imshow(x.cpu().detach().permute(1,2,0))
    return ax


def show_imgs(xb: t.Tensor, ax=None):
    """Arranges and displays a batch of image tensors (B, C, H, W) in a grid."""
    nrow = int(xb.shape[0]**0.5)
    xb = make_grid(xb, nrow=nrow, pad_value=1)
    return show_img(xb, ax)


# Selects CUDA if available, otherwise defaults to CPU.
default_device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def to_device(x, device=default_device):
    """Recursively moves tensors or collections (lists, tuples, dicts) of tensors to a device."""
    if isinstance(x, t.Tensor): return x.to(device)
    if isinstance(x, (tuple, list)): return [to_device(el, device) for el in x]
    if isinstance(x, dict): return {k:to_device(v) for k,v in x.items()}
    return x