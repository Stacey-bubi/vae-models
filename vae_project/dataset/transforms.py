from ..imports import *
from ..utils import noop


class CollateWithTransforms:
    """A collate function that applies batch transforms"""

    def __init__(self, transforms: list = None):
        self.transforms = transforms if transforms is not None else []

    def __call__(self, batch):
        batch = torch.utils.data.default_collate(batch)
        for t in self.transforms:
            batch = t(batch)
        return batch


class ToBatchTransform:
    """Returns batch transform that optionally applies `transform_x` to x and `transform_y` to y"""

    def __init__(self, transform_x: Callable = noop, transform_y: Callable = noop):
        self.tfm_x, self.tfm_y = transform_x, transform_y

    def __call__(self, batch):
        x, y = batch
        return self.tfm_x(x), self.tfm_y(y)


class Colorize:
    """Colorizes batch of single channel MNIST images based on labels or randomly."""

    def __init__(self, color_type: Literal["random", "label", "random_discrete"] = "label", n_classes=10):
        """Initializes the colorization transform.

        Args:
            color_type (str): The colorization strategy.
                - "label": Colors are based on the image label.
                - "random": Colors are chosen from a continuous random distribution.
                - "random_discrete": Colors are chosen randomly from a fixed set
                  of equally spaced hues.
            n_classes (int): The total number of classes ("label" mode) or number of colors in the discrete palette ("random_discrete" mode).
                Required for "random_discrete" mode.
        """
        self.n_classes, self.color_type = n_classes, color_type
        if self.color_type == "random_discrete":
            self.discrete_hues = t.linspace(0, 6, steps=self.n_classes + 1)

    def _colorize_batch(self, images: t.Tensor, hues: t.Tensor):
        h = hues.unsqueeze(1)
        # convert h to rgb
        c = t.cat([(h - 3).abs() - 1, 2 - (h - 2).abs(), 2 - (h - 4).abs()], dim=1).clamp(0, 1)
        return images * c.view(-1, 3, 1, 1)

    def __call__(self, batch: tuple[t.Tensor, t.Tensor]):
        images, labels = batch

        if self.color_type == "label":
            hues = labels.float() / self.n_classes * 6
        elif self.color_type == "random":
            hues = t.rand(images.shape[0], device=images.device) * 6
        elif self.color_type == "random_discrete":
            indices = t.randint(0, self.n_classes + 1, (images.shape[0],), device=images.device)
            hues = self.discrete_hues.to(images.device)[indices]
        else:
            return images, labels

        colored_imgs = self._colorize_batch(images, hues)
        return colored_imgs, labels
