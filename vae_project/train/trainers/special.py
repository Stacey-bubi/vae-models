from vae_project.models import VAE, IWAE, VampPriorVAE
from ...imports import *
from .base import BaseTrainer
from ...utils import to_device
from ..losses import vamp_prior_elbo_loss, iwae_loss, elbo_loss


class Trainer(BaseTrainer):
    """VAE trainer

    To use this trainer, ensure your:
    - `model`'s forward pass returns `(reconstruction, mu, log_var)`.
    - `loss_func` accepts `(preds, original_input, mu, log_var, beta)`.
      The `beta` value for the KL divergence weight can be controlled by a hook
      by setting `trainer.beta` during training (e.g., for beta annealing).
    """

    model: VAE

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

    model: IWAE

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

    def get_loss(self):
        """Calculates the IWAE loss using importance weighting."""
        if self.loss_func is None:
            self.loss_func = iwae_loss
        return self.loss_func(self.preds, self.xb, self.z, self.mu, self.log_var, getattr(self, "beta", 1), self.model.recon_dist)

    def predict(self, xb, training: bool = None):
        """Runs a forward pass on the IWAE model with importance samples."""
        xb = to_device(xb, self.device)
        if training is None:
            training = self.training
        K = self.K_train if training else self.K_eval
        self.preds, self.z, self.mu, self.log_var = self.model(xb, K=K)
        return self.preds


class VampPriorTrainer(BaseTrainer):
    """
    Trainer specifically for VampPrior VAE with colored MNIST.

    This trainer handles the specific requirements of VampPrior training:
    - Computing log p(z) under VampPrior
    - Managing the VampPrior-specific loss function
    - Supporting beta annealing for KL term
    """

    model: VampPriorVAE

    def predict(self, xb: torch.Tensor):
        xb = to_device(xb, self.device)

        self.preds, self.mu, self.log_var = self.model(xb)
        self.z = self.model.z
        self.log_p_z = self.model.prior.log_p_z(self.z)
        return self.preds

    def get_loss(self) -> torch.Tensor:
        if self.loss_func is None:
            self.loss_func = vamp_prior_elbo_loss
        return self.loss_func(
            self.preds,
            self.xb,
            self.mu,
            self.log_var,
            self.log_p_z,
            beta=getattr(self, "beta", 1),
            recon_dist=self.model.recon_dist,
        )
