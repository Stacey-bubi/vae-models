from tqdm.auto import tqdm
from ..imports import *
from ..utils import default_device


class Trainer:
    def __init__(self, model, train_dl, val_dl, optim, loss_func, epochs=10, hooks=None, verbose=True, device=default_device):
        """Helper class that contains training and evaluation loop."""
        self.model, self.train_dl, self.val_dl, self.opt, self.loss_func = model, train_dl, val_dl, optim, loss_func
        self.epochs, self.hooks = epochs, hooks if hooks else []
        self.verbose = verbose
        self.device = device
        self.model.to(self.device)

    def _call_hook(self, method_name):
        for hook in self.hooks:
            getattr(hook, method_name, lambda trainer: None)(self)

    def _one_batch(self):
        self.xb, self.yb = self.batch[0].to(self.device), self.batch[1]  # yb not used in VAE, but good practice
        self._call_hook("begin_step")
        self.preds, self.mu, self.log_var = self.model(self.xb)
        self._call_hook("after_pred")
        self.loss = self.loss_func(self.preds, self.xb, self.mu, self.log_var)
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
            if self.verbose: dl_iter.set_postfix(avg_loss=f'{running_loss/(self.batch_idx):.4f}')

    def fit(self):
        self._call_hook("begin_fit")
        for self.epoch in tqdm(range(self.epochs)):
            self._call_hook("begin_epoch")
            self._one_epoch(self.train_dl, training=True, desc=f"Epoch {self.epoch+1}/{self.epochs} [Train]")
            with t.no_grad():
                self._one_epoch(self.val_dl, training=False, desc=f"Epoch {self.epoch+1}/{self.epochs} [Valid]")
            self._call_hook("after_epoch")
        self._call_hook("after_fit")
