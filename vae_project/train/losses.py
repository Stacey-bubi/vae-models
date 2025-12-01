from ..imports import *


def kl_loss(mu: t.Tensor, log_var: t.Tensor):
    """Return KL divergence between N(`mu`, exp(`log_var`)) and N(0,I)"""
    return -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]


def recon_loss(recon_x: t.Tensor, x: t.Tensor, recon: str = "bce"):
    '''Reconstruction loss'''
    if recon.lower() == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    elif recon.lower() == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    else:
        raise ValueError(f"Unknown recon loss: {recon}")
    return recon_loss / x.shape[0]


def elbo_loss(recon_x: t.Tensor, x: t.Tensor, mu: t.Tensor, log_var: t.Tensor, beta: float = 1.0, recon: str = "bce") -> t.Tensor:
    """Compute ELBO = reconstruction loss + beta * KL.

    - recon == 'bce': binary cross-entropy on [0,1] images (expects recon_x in [0,1]).
    - recon == 'mse': mean squared error (useful for continuous targets).

    Both terms are averaged per sample (sum over dims, mean over batch).
    """
    return recon_loss(recon_x, x, recon) + beta * kl_loss(mu, log_var)
