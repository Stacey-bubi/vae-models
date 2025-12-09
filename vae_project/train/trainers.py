from typing import Type, TypeVar

from vae_project.train.losses import elbo_loss, iwae_loss
from ..imports import *
from ..utils import default_device, to_device

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
            self.loss_t.backward()
            self._call_hook("after_backward")
            self.opt.step()
            self.opt.zero_grad()
            self.step += 1
        self._call_hook("after_step")

    def _one_epoch(self):
        """Run single epoch"""
        for self.batch_idx, self.batch in enumerate(self.dl):
            self._one_batch()

    def fit(self):
        """Starts the training and validation loops for the specified number of epochs."""
        self.n_steps = len(self.train_dl) * self.epochs
        self.step = 0
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
        if self.loss_func is None:
            self.loss_func = elbo_loss
        return self.loss_func(self.preds, self.xb, self.mu, self.log_var, getattr(self, "beta", 1), self.model.recon_dist)

    def predict(self, xb):
        """Runs a forward pass on the VAE model and stores its outputs."""
        xb = to_device(xb, self.device)
        self.preds, self.mu, self.log_var = self.model(xb)
        return self.preds


class TrainerIWAE(BaseTrainer):
    """Importance-Weighted Autoencoder (IWAE) trainer.

    To use this trainer, ensure your:
    - `model` is an IWAE instance with `forward(x, K)` method.
    - `loss_func` accepts `(recon, x, z, mu, log_var, K, beta)` arguments.
      The `beta` value can be controlled by a hook (e.g., for beta annealing).
      The K value should be set via the trainer's K_train and K_eval attributes.
    """

    def __init__(self, K_train: int = 5, K_eval: int = 50, **kwargs):
        """Initialize IWAE trainer.

        Args:
            K_train: Number of importance samples during training
            K_eval: Number of importance samples during evaluation (typically larger)
            ... (Same as for `BaseTrainer`)
        """
        super().__init__(**kwargs)
        self.K_train = K_train
        self.K_eval = K_eval
        if self.loss_func is None:
            self.loss_func = iwae_loss

    def get_loss(self):
        """Calculates the IWAE loss using importance weighting."""
        return self.loss_func(self.preds, self.xb, self.z, self.mu, self.log_var, getattr(self, "beta", 1), self.model.recon_dist)

    def predict(self, xb, training: bool = None):
        """Runs a forward pass on the IWAE model with importance samples."""
        xb = to_device(xb, self.device)
        if training is None:
            training = self.training
        K = self.K_train if training else self.K_eval
        self.preds, self.z, self.mu, self.log_var = self.model(xb, K=K)
        return self.preds
