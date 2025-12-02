from collections import defaultdict
from vae_project.imports import *
from vae_project.train.losses import *
from vae_project.train import Trainer
from tqdm.auto import tqdm
from .base import BaseHook

try:
    import trackio as tio
except:
    pass


class MetricsHook(BaseHook):
    """A hook to collect and plot training and validation loss."""

    def __init__(self, verbose=True, recon_loss="bce",  use_trackio=False, **trackio_kwargs):
        self.use_trackio, self.verbose, self.recon_loss = use_trackio, verbose, recon_loss
        self.metrics = defaultdict(list)
        self.val_losses = []
        if use_trackio:
            tio.init('vae', **trackio_kwargs)

    def before_fit(self, trainer):
        self.n_train, self.n_valid = len(trainer.train_dl), len(trainer.valid_dl)

    def before_valid(self, trainer):
        self.val_batch_metrics = defaultdict(list)

    def after_loss(self, trainer: Trainer):
        data = {}
        prefix = "train" if trainer.training else "valid"
        data[f"{prefix}_kl"] = kl_loss(trainer.mu, trainer.log_var).item()
        data[f"{prefix}_recon"] = recon_loss(trainer.preds, trainer.xb, self.recon_loss).item()
        data[f"{prefix}_elbo"] = trainer.loss
        data["beta"] = getattr(trainer, "beta", 1)

        # for validation aggregate each epoch, then store
        metrics_dict = self.val_batch_metrics if not trainer.training else self.metrics
        for k, v in data.items():
            metrics_dict[k].append(v)

        if self.use_trackio:
            tio.log(data, trainer.step)

    def after_epoch(self, trainer: Trainer):
        # store validation metrics
        for k, v in self.val_batch_metrics.items():
            self.metrics[k].append(np.mean(v))

        train_loss = np.mean(self.metrics["train_elbo"][-self.n_train :])
        val_loss = self.metrics["valid_elbo"][-1]
        if self.verbose:
            print(f"Epoch {trainer.epoch+1}/{trainer.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def after_fit(self, trainer):
        if self.use_trackio:
            tio.finish()

    def plot_loss(self, axes=None):
        """Plots graphs for ELBO, KL, and Reconstruction losses."""
        metrics_to_plot = ["elbo", "kl", "recon"]
        if not axes:
            axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)))[1]
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, m in zip(axes, metrics_to_plot):
            ax.plot(self.metrics[f"train_{m}"], label=f"Train {m.title()}")
            val_x = np.arange(1, len(self.metrics[f"valid_{m}"]) + 1) * self.n_train - 1
            ax.plot(val_x, self.metrics[f"valid_{m}"], "o-", label=f"Valid {m.title()}")
            ax.set_title(f"{m.title()}")
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Batch / Step")
        plt.tight_layout()
        plt.show()


class ProgressBarHook(BaseHook):
    """A hook to display progress bars for epochs and batches."""

    def before_fit(self, trainer: Trainer):
        self.epoch_bar = tqdm(range(trainer.epochs), desc="Epoch")

    def before_epoch(self, trainer: Trainer):
        self.losses = []
        trainer.dl = self.bar = tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Train]", leave=False)

    def before_valid(self, trainer: Trainer):
        self.losses = []
        trainer.dl = self.bar = tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Valid]", leave=False)

    def after_step(self, trainer: Trainer):
        self.losses.append(trainer.loss)
        self.bar.set_postfix(loss=f"{np.mean(self.losses):.4f}")

    def after_epoch(self, trainer: Trainer):
        self.epoch_bar.update(1)

    def after_fit(self, _):
        self.epoch_bar.close()
