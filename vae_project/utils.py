import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch as t


def show_img(x: t.Tensor, ax=None):
    if ax is None: ax = plt.subplot()
    ax.imshow(x.cpu().detach().permute(1,2,0))
    return ax


def show_imgs(xb: t.Tensor, ax=None):
    nrow = int(xb.shape[0]**0.5)
    xb = make_grid(xb, nrow=nrow, pad_value=1)
    return show_img(xb, ax)


default_device = t.device('cuda' if t.cuda.is_available() else 'cpu')