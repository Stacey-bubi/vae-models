from collections import defaultdict
from vae_project.imports import *
from vae_project.train.losses import *
from vae_project.train import Trainer
from .base import BaseHook

try:
    import trackio as tio
except:
    pass


class BaseMetricsHook(BaseHook):
    """An abstract hook to collect, aggregate, and plot training/validation metrics."""

    def __init__(self, verbose=True, use_trackio=False, **trackio_kwargs):
        self.use_trackio, self.verbose = use_trackio, verbose
        self.metrics = defaultdict(list)
        if use_trackio:
            try:
                tio.init("vae", **trackio_kwargs)
            except:
                print("Failed to init trackio")
                self.use_trackio = False

    def _get_batch_metrics(self, trainer, prefix: str = Literal["train", "valid"]) -> dict:
        """Subclasses MUST implement this to return a dictionary of metrics for the current step."""
        raise NotImplementedError

    def before_fit(self, trainer):
        self.n_train, self.n_valid = len(trainer.train_dl), len(trainer.valid_dl)

    def before_valid(self, trainer):
        self.val_batch_metrics = defaultdict(list)

    def after_loss(self, trainer):
        prefix = "train" if trainer.training else "valid"
        data = self._get_batch_metrics(trainer, prefix)

        metrics_dict = self.val_batch_metrics if not trainer.training else self.metrics
        for k, v in data.items():
            metrics_dict[k].append(v)
        if self.use_trackio:
            tio.log(data, trainer.step)

    def after_epoch(self, trainer):
        for k, v in self.val_batch_metrics.items():
            self.metrics[k].append(np.mean(v))
        train_loss = np.mean(self.metrics["train_loss"][-self.n_train :])
        val_loss = self.metrics["valid_loss"][-1]
        if self.verbose:
            print(f"Epoch {trainer.epoch+1}/{trainer.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def after_fit(self, trainer):
        if self.use_trackio:
            tio.finish()

    def plot_loss(self, axes=None):
        """Plots graphs for all available metrics (e.g., loss, kl, recon)."""
        train_keys = sorted([k for k in self.metrics if k.startswith("train_")])
        metrics_to_plot = [k.replace("train_", "") for k in train_keys]

        if not axes:
            _, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 4 * len(metrics_to_plot)), squeeze=False)
        axes = axes.flatten()

        for i, m in enumerate(metrics_to_plot):
            ax = axes[i]
            ax.plot(self.metrics[f"train_{m}"], label=f"Train {m.title()}")
            if f"valid_{m}" in self.metrics:
                val_x = np.arange(1, len(self.metrics[f"valid_{m}"]) + 1) * self.n_train - 1
                ax.plot(val_x, self.metrics[f"valid_{m}"], "o-", label=f"Valid {m.title()}")
            ax.set_title(m.replace("_", " ").title())
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Batch / Step")
        plt.tight_layout()
        plt.show()


class MetricsHook(BaseMetricsHook):
    """Metrics hook that only logs the total loss."""

    def _get_batch_metrics(self, trainer, prefix: str) -> dict:
        return {f"{prefix}_loss": trainer.loss}


class VAEMetricsHook(MetricsHook):
    """Metrics hook for standard VAEs that logs reconstruction and KL terms."""

    def __init__(self, recon_loss_fn="bce", **kwargs):
        super().__init__(**kwargs)
        self.recon_loss_fn = recon_loss_fn

    def _get_batch_metrics(self, trainer, prefix: str) -> dict:
        data = super()._get_batch_metrics(trainer, prefix)
        return {
            **data,
            f"{prefix}_kl": kl_loss(trainer.mu, trainer.log_var).item(),
            f"{prefix}_recon": recon_loss(trainer.preds, trainer.xb, self.recon_loss_fn).item(),
            "beta": getattr(trainer, "beta", 1.0),
        }
