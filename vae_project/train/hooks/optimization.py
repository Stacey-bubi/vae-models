from vae_project.imports import *
from vae_project.train import Trainer
from .base import BaseHook


class BetaSchedulerHook(BaseHook):
    """Schedules the beta value for KL divergence annealing."""

    def __init__(self, n_steps=None, start=0.0, end=1.0):
        self.n_steps, self.start, self.end = n_steps, start, end

    def before_fit(self, trainer: Trainer):
        if self.n_steps is None:
            self.n_steps = trainer.n_steps
        self.increment = (self.end - self.start) / self.n_steps
        trainer.beta = self.start

    def after_step(self, trainer: Trainer):
        if trainer.training and trainer.step < self.n_steps:
            trainer.beta = min(self.end, trainer.beta + self.increment)


class GradClipHook(BaseHook):
    """Hook to clip gradient"""

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_backward(self, trainer: Trainer):
        nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)
