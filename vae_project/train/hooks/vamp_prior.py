from ..trainers import VampPriorTrainer
from ...imports import *
from ...utils import default_device
from torchvision.utils import make_grid, save_image

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
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.opt.state_dict(),
                    "beta": getattr(self, "beta", 1),
                },
                checkpoint_path,
            )
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
