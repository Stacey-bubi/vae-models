from ..imports import *
from .trainer import Trainer


class BaseHook:
    """Base class for hooks. Hooks can interact with the Trainer at various points."""

    def begin_fit(self, trainer):
        pass

    def begin_epoch(self, trainer):
        pass

    def begin_step(self, trainer):
        pass

    def after_pred(self, trainer):
        pass

    def after_loss(self, trainer):
        pass

    def after_backward(self, trainer):
        pass

    def after_step(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def after_fit(self, trainer):
        pass


class MetricsHook(BaseHook):
    """A hook to collect and plot training and validation loss."""

    def __init__(self):
        self.metrics = {"train_loss": [], "val_loss": [], "beta": []}
        self.val_losses = []

    def after_step(self, trainer: Trainer):
        if trainer.training:
            self.metrics["train_loss"].append(trainer.loss.item())
            self.metrics["beta"].append(getattr(trainer, "beta", 1))
        else:
            self.val_losses.append(trainer.loss.item())

    def after_epoch(self, trainer: Trainer):
        train_loss = sum(self.metrics["train_loss"][-len(trainer.train_dl) :]) / len(trainer.train_dl)
        val_loss = np.array(self.val_losses).mean()
        self.metrics["val_loss"].append(val_loss)
        print(f"Epoch {trainer.epoch+1}/{trainer.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        self.val_losses = []

    def plot_loss(self):
        """Plots loss function graphs."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_loss"], label="Training Loss")
        val_points = [
            i * len(self.metrics["train_loss"]) // len(self.metrics["val_loss"]) for i in range(len(self.metrics["val_loss"]))
        ]
        plt.plot(val_points, self.metrics["val_loss"], label="Validation Loss", marker="o", linestyle="dashed")
        plt.title("Training and Validation Loss")
        plt.xlabel("Batch/Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()


class BetaSchedulerHook(BaseHook):
    """Schedules the beta value for KL divergence annealing."""

    def __init__(self, n_steps=None, start=0.0, end=1.0):
        self.n_steps, self.start, self.end = n_steps, start, end

    def begin_fit(self, trainer: Trainer):
        self.step_count = 0
        if self.n_steps is None:
            self.n_steps = len(trainer.train_dl) * trainer.epochs
        self.increment = (self.end - self.start) / self.n_steps
        trainer.beta = self.start

    def after_step(self, trainer: Trainer):
        if trainer.training and self.step_count < self.n_steps:
            trainer.beta = min(self.end, trainer.beta + self.increment)
            self.step_count += 1


class GradClipHook(BaseHook):
    '''Hook to clip gradient'''
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_backward(self, trainer: Trainer):
        nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)
