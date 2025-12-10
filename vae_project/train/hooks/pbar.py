import numpy as np
from tqdm.auto import tqdm as auto_tqdm
from tqdm import tqdm
from .base import BaseHook


class ProgressBarHook(BaseHook):
    """A hook to display progress bars for epochs and batches."""
    def __init__(self, text_only: bool = False):
        self.tqdm = tqdm if text_only else auto_tqdm
    

    def before_fit(self, trainer):
        self.epoch_bar = self.tqdm(range(trainer.epochs), desc="Epoch")

    def before_epoch(self, trainer):
        self.losses = []
        trainer.dl = self.bar = self.tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Train]", leave=False)

    def before_valid(self, trainer):
        self.losses = []
        trainer.dl = self.bar = self.tqdm(trainer.dl, desc=f"Epoch {trainer.epoch+1}/{trainer.epochs} [Valid]", leave=False)

    def after_step(self, trainer):
        self.losses.append(trainer.loss)
        self.bar.set_postfix(loss=f"{np.mean(self.losses):.4f}")

    def after_epoch(self, trainer):
        self.epoch_bar.update(1)

    def after_fit(self, _):
        self.epoch_bar.close()
