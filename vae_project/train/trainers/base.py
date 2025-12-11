from typing import Type, TypeVar, Callable
from ...imports import *
from ...utils import default_device, to_device

T = TypeVar("T")


class BaseTrainer:
    """
    Helper class that contains training and evaluation loop.

    This class provides a generic framework for training. To adapt it for a specific
    model or task, you can extend its functionality in two primary ways:

    1. Inheritance:
       Create a new class that inherits from `BaseTrainer`. You MUST implement
       the following methods:
       - `predict(self, xb)`: Defines how the model processes an input batch `xb`.
         The results should be stored as instance attributes (e.g., `self.preds`)
         to be used by the loss function.
       - `get_loss(self)`: Defines how the loss is calculated using the outputs
         from `predict()` and the ground truth `self.yb`.

    2. Hooks (Callbacks):
        You can also add call other functions or add/modify attributes during training by using callbacks. You have to create
       class with methods corresponding to training stages. The available hook points
       are: `begin_fit`, `after_fit`, `begin_epoch`, `after_epoch`, `begin_step`,
       `after_step`, `after_pred`, `after_loss`, and `after_backward`. Pass a list
       of hook instances or single hook to the `hooks` parameter during initialization.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dl,
        valid_dl,
        optim: t.optim.Optimizer,
        loss_func: Callable = None,
        epochs=10,
        hooks=None,
        device=default_device,
    ):
        self.model, self.train_dl, self.valid_dl, self.opt, self.loss_func = model, train_dl, valid_dl, optim, loss_func
        self.epochs, self.hooks = epochs, hooks if hooks else []
        self.device = device
        self.model.to(self.device)
        self.step = 0

    def _call_hook(self, method_name):
        for hook in self.hooks:
            getattr(hook, method_name, lambda trainer: None)(self)

    def get_loss(self) -> torch.Tensor:
        """Calculates the loss for the current batch. Must be implemented by a subclass."""
        raise NotImplementedError

    def predict(self, xb):
        """Performs a forward pass on the model. Must be implemented by a subclass."""
        raise NotImplementedError

    def _one_batch(self):
        """Process single batch forward, optionally with backward"""
        self.xb, self.yb = to_device(self.batch, self.device)
        self._call_hook("before_step")
        self.predict(self.xb)
        self._call_hook("after_pred")
        self.loss_t = self.get_loss()
        self.loss = self.loss_t.item()
        self._call_hook("after_loss")
        if self.model.training:
            self.opt.zero_grad()
            self.loss_t.backward()
            self._call_hook("after_backward")
            self.opt.step()
            self.step += 1
        self._call_hook("after_step")

    def _one_epoch(self):
        """Run single epoch"""
        for self.batch_idx, self.batch in enumerate(self.dl):
            self._one_batch()

    def fit(self):
        """Starts the training and validation loops for the specified number of epochs."""
        self.n_steps = len(self.train_dl) * self.epochs
        self._call_hook("before_fit")
        for self.epoch in range(self.epochs):
            # Train
            self.model.train()
            self.training, self.dl = True, self.train_dl
            self._call_hook("before_epoch")
            self._one_epoch()

            # Validation
            self.model.eval()
            self.training, self.dl = False, self.valid_dl
            self._call_hook("before_valid")
            with torch.no_grad():
                self._one_epoch()
            self._call_hook("after_epoch")
        self._call_hook("after_fit")

    def get_hook(self, cls: Type[T]) -> T:
        for h in self.hooks:
            if isinstance(h, cls):
                return h
        raise KeyError(f"Hook {cls} not found")
