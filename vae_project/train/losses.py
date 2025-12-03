from ..imports import *


def kl_loss(mu: t.Tensor, log_var: t.Tensor):
    """Return KL divergence between N(`mu`, exp(`log_var`)) and N(0,I)"""
    return -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]


def recon_loss(recon_x: t.Tensor, x: t.Tensor, recon: str = "bce"):
    """Reconstruction loss"""
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


def iwae_loss(
    recon_xs: t.Tensor, x: t.Tensor, z: t.Tensor, mu: t.Tensor, log_var: t.Tensor, beta: float = 1.0, recon: str = "bce"
) -> t.Tensor:
    """Computes the IWAE loss, which is the negative of the IWAE log-likelihood bound."""
    B, K = z.shape[:2]

    x_exp = x.unsqueeze(1).expand_as(recon_xs)
    if recon.lower() == "bce":
        nll = F.binary_cross_entropy(recon_xs, x_exp, reduction="none")
    elif recon.lower() == "mse":
        nll = F.mse_loss(recon_xs, x_exp, reduction="none")
    else:
        raise ValueError(f"Unknown recon loss: {recon}")
    log_p_x_z = -nll.view(B, K, -1).sum(-1)  # Sum over all pixel/channel dims

    # Compute log p(z) - Prior log-likelihood
    log_p_z = -0.5 * (z**2).sum(-1)  # Sum over latent_dim

    # Compute log q(z|x) - Posterior log-likelihood
    mu_exp, log_var_exp = mu.unsqueeze(1), log_var.unsqueeze(1)
    log_q_z_x = -0.5 * (((z - mu_exp) ** 2 / log_var_exp.exp()) + log_var_exp).sum(-1)

    log_w = log_p_x_z + beta * (log_p_z - log_q_z_x)  # Shape: [B, K]

    # Log-sum-exp trick
    log_w_max = log_w.max(dim=1, keepdim=True).values
    log_mean_w = (log_w - log_w_max).exp().mean(dim=1).log() + log_w_max.squeeze(1)

    return -log_mean_w.mean()
