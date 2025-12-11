from typing import Type, TypeVar, List, Optional, Callable
import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid, save_image
from ..imports import *
from ..utils import default_device, to_device
from .losses import vamp_prior_elbo_loss, iwae_loss, elbo_loss

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


class VampPriorTrainer(BaseTrainer):
    """
    Trainer specifically for VampPrior VAE with colored MNIST.
    
    This trainer handles the specific requirements of VampPrior training:
    - Computing log p(z) under VampPrior
    - Managing the VampPrior-specific loss function
    - Supporting beta annealing for KL term
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dl,
        valid_dl,
        optim: torch.optim.Optimizer,
        loss_func: Optional[Callable] = None,
        epochs: int = 10,
        hooks: Optional[List] = None,
        device: str = default_device,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 10000
    ):
        """
        Initialize VampPrior trainer for colored MNIST.
        
        Args:
            beta_start: Initial beta value (start of annealing)
            beta_end: Final beta value (end of annealing)
            beta_anneal_steps: Number of steps over which to anneal beta
        """
        if loss_func is None:
            loss_func = vamp_prior_elbo_loss
        
        super().__init__(
            model=model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            optim=optim,
            loss_func=loss_func,
            epochs=epochs,
            hooks=hooks,
            device=device
        )
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.current_step = 0
        
        # Set initial beta
        self.beta = beta_start if beta_anneal_steps > 0 else beta_end

    def _update_beta(self):
        """Update beta value based on current training step."""
        if self.beta_anneal_steps > 0 and self.current_step <= self.beta_anneal_steps:
            progress = self.current_step / self.beta_anneal_steps
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            self.beta = self.beta_end

    def predict(self, xb: torch.Tensor):
        """
        Forward pass that computes all required values for VampPrior loss.
        
        Args:
            xb: Input batch [batch_size, channels, height, width]
        """
        xb = to_device(xb, self.device)
        
        # Forward pass through VampPrior VAE
        recon_x, mu, log_var = self.model(xb)
        
        # Get z from model (stored during forward pass)
        z = self.model.z
        
        # Compute log p(z) under VampPrior
        log_p_z = self.model.prior.log_p_z(z)
        
        # Store all values needed for loss computation
        self.preds = recon_x
        self.mu = mu
        self.log_var = log_var
        self.log_p_z = log_p_z
        self.z = z

    def get_loss(self) -> torch.Tensor:
        """
        Compute VampPrior ELBO loss with current beta value.
        
        This method handles beta annealing automatically.
        """
        # Update beta for annealing
        if self.training:
            self._update_beta()
            self.current_step += 1
        
        return self.loss_func(
            self.preds, 
            self.xb, 
            self.mu, 
            self.log_var, 
            self.log_p_z,
            beta=self.beta,
            recon_dist=self.model.recon_dist
        )

    def get_metrics(self) -> dict:
        """Get training metrics including current beta value."""
        return {'beta': self.beta}


class VampPriorHook:
    """
    Hook for VampPrior training that provides additional functionality for colored MNIST.
    
    This hook can:
    - Save pseudo-inputs during training
    - Generate samples from VampPrior
    - Save reconstructions of real images
    - Monitor KL divergence components
    - Save model checkpoints
    """
    
    def __init__(self, save_dir: str = "./results", save_every: int = 10, device: str = default_device):
        self.save_dir = save_dir
        self.save_every = save_every
        self.device = device
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "pseudo_inputs"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "reconstructions"), exist_ok=True)
        self.last_saved_epoch = -1  # Track last saved epoch
    
    def after_epoch(self, trainer: VampPriorTrainer):
        """Called after each epoch."""
        epoch = trainer.epoch
        model = trainer.model
        
        if (epoch + 1) % self.save_every == 0 and epoch != self.last_saved_epoch:
            self.last_saved_epoch = epoch
            
            # Save pseudo-inputs visualization
            self._save_pseudo_inputs(model, epoch)
            
            # Generate and save samples from VampPrior
            self._save_samples(model, epoch)
            
            # Save reconstructions of validation batch
            self._save_reconstructions(model, trainer.valid_dl, epoch)
            
            # Save model checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"vampprior_vae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.opt.state_dict(),
                'beta': trainer.beta,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    def _save_pseudo_inputs(self, model: nn.Module, epoch: int):
        """Save visualization of pseudo-inputs for colored MNIST."""
        try:
            with torch.no_grad():
                pseudo_inputs = model.prior.get_pseudo_inputs().cpu()
                # Take first 64 pseudo-inputs for visualization
                pseudo_inputs = pseudo_inputs[:64]
                
                # Create grid visualization - handle RGB channels
                grid = make_grid(pseudo_inputs, nrow=8, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.save_dir, "pseudo_inputs", f"pseudo_inputs_epoch_{epoch+1}.png"))
        except Exception as e:
            print(f"⚠ Failed to save pseudo-inputs: {str(e)}")
    
    def _save_samples(self, model: nn.Module, epoch: int):
        """Generate and save samples from VampPrior for colored MNIST."""
        try:
            with torch.no_grad():
                # Generate 64 samples
                samples = model.sample(n=64, device=self.device).cpu()
                
                # Create grid visualization
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.save_dir, "samples", f"samples_epoch_{epoch+1}.png"))
        except Exception as e:
            print(f"⚠ Failed to save samples: {str(e)}")
    
    def _save_reconstructions(self, model: nn.Module, valid_dl, epoch: int):
        """Save reconstructions of validation batch for colored MNIST."""
        try:
            with torch.no_grad():
                # Get a batch from validation set
                batch = next(iter(valid_dl))[0]
                batch = batch.to(self.device)[:32]  # Take first 32 images
                
                # Get reconstructions
                reconstructions = model(batch)[0].cpu()
                originals = batch.cpu()
                
                # Create comparison grid
                comparison = torch.cat([originals, reconstructions], dim=0)
                grid = make_grid(comparison, nrow=8, normalize=True, value_range=(0, 1))
                save_image(grid, os.path.join(self.save_dir, "reconstructions", f"reconstructions_epoch_{epoch+1}.png"))
        except Exception as e:
            print(f"⚠ Failed to save reconstructions: {str(e)}")