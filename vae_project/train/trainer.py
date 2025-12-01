from tqdm.auto import tqdm
from ..imports import *
from ..utils import default_device, to_device


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
        model,
        train_dl,
        val_dl,
        optim: t.optim.Optimizer,
        loss_func: Callable,
        epochs=10,
        hooks=None,
        verbose=True,
        device=default_device,
    ):
        self.model, self.train_dl, self.val_dl, self.opt, self.loss_func = model, train_dl, val_dl, optim, loss_func
        self.epochs, self.hooks = epochs, hooks if hooks else []
        self.verbose = verbose
        self.device = device
        self.model.to(self.device)

    def _call_hook(self, method_name):
        for hook in self.hooks:
            getattr(hook, method_name, lambda trainer: None)(self)

    def get_loss(self):
        """Calculates the loss for the current batch. Must be implemented by a subclass."""
        raise NotImplementedError

    def predict(self, xb):
        """Performs a forward pass on the model. Must be implemented by a subclass."""
        raise NotImplementedError

    def _one_batch(self):
        self.xb, self.yb = to_device(self.batch, self.device)
        self._call_hook("begin_step")
        self.predict(self.xb)
        self._call_hook("after_pred")
        self.loss = self.get_loss()
        self._call_hook("after_loss")
        if self.model.training:
            self.loss.backward()
            self._call_hook("after_backward")
            self.opt.step()
            self.opt.zero_grad()
        self._call_hook("after_step")

    def _one_epoch(self, dl, training, desc):
        self.model.training = training
        self.dl = dl
        dl_iter = tqdm(self.dl, desc=desc, leave=False) if self.verbose else self.dl
        running_loss = 0
        for self.batch_idx, self.batch in enumerate(dl_iter):
            self._one_batch()
            running_loss += self.loss
            if self.verbose:
                dl_iter.set_postfix(avg_loss=f"{running_loss/(self.batch_idx+1):.4f}")

    def fit(self):
        """Starts the training and validation loops for the specified number of epochs."""
        self._call_hook("begin_fit")
        for self.epoch in tqdm(range(self.epochs)):
            self._call_hook("begin_epoch")
            self._one_epoch(self.train_dl, training=True, desc=f"Epoch {self.epoch+1}/{self.epochs} [Train]")
            with t.no_grad():
                self._one_epoch(self.val_dl, training=False, desc=f"Epoch {self.epoch+1}/{self.epochs} [Valid]")
            self._call_hook("after_epoch")
        self._call_hook("after_fit")


class Trainer(BaseTrainer):
    """VAE trainer

    To use this trainer, ensure your:
    - `model`'s forward pass returns `(reconstruction, mu, log_var)`.
    - `loss_func` accepts `(preds, original_input, mu, log_var, beta)`.
      The `beta` value for the KL divergence weight can be controlled by a hook
      by setting `trainer.beta` during training (e.g., for beta annealing).
    """

    def get_loss(self):
        """Calculates the VAE loss, combining reconstruction and KL divergence."""
        return self.loss_func(self.preds, self.xb, self.mu, self.log_var, getattr(self, "beta", 1))

    def predict(self, xb):
        """Runs a forward pass on the VAE model and stores its outputs."""
        self.preds, self.mu, self.log_var = self.model(xb)
        return self.preds, self.mu, self.log_var
