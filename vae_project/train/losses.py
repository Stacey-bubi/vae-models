from ..imports import *
import math

def kl_loss(mu: t.Tensor, log_var: t.Tensor):
    """Return KL divergence between N(`mu`, exp(`log_var`)) and N(0,I)"""
    return -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]

def recon_loss(recon_x: t.Tensor, x: t.Tensor, recon: str = "bce"):
    """Reconstruction loss"""
    if recon.lower() == "bce":
        # Ensure inputs are in [0,1] range for BCE
        assert 0 <= x.min() <= x.max() <= 1, "Input must be in [0,1] range for BCE loss"
        return F.binary_cross_entropy(recon_x, x, reduction="sum") / x.shape[0]
    elif recon.lower() == "mse":
        return F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
    else:
        raise ValueError(f"Unknown recon loss: {recon}")

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
    
def vamp_prior_kl_loss(mu: torch.Tensor, log_var: torch.Tensor, log_p_z: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between q(z|x) and VampPrior p(z).
    
    This is the core KL term for VampPrior VAEs.
    
    Args:
        mu: Posterior means [batch_size, latent_dim]
        log_var: Posterior log variances [batch_size, latent_dim]
        log_p_z: Log prior probability under VampPrior [batch_size]
    
    Returns:
        KL divergence averaged over batch
    """
    # Compute log q(z|x) for samples from posterior
    latent_dim = mu.shape[-1]
    
    # E_q[log q(z|x)] = -0.5 * (latent_dim * log(2π) + log|Σ| + d)
    # where d = latent_dim for standard Gaussian, but we compute exactly
    log_q_z_x = -0.5 * (
        log_var.sum(-1) +  # log|Σ|
        latent_dim * math.log(2 * math.pi) +  # latent_dim * log(2π)
        latent_dim  # The expectation of squared Mahalanobis distance
    )
    
    # KL = E_q[log q(z|x) - log p(z)]
    kl = (log_q_z_x - log_p_z).mean()
    
    # Ensure non-negative for numerical stability
    return torch.clamp(kl, min=0.0)

def vamp_prior_elbo_loss(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    log_var: torch.Tensor, 
    log_p_z: torch.Tensor,
    beta: float = 1.0, 
    recon_dist: Literal["bce", "mse"] = "bce"
) -> torch.Tensor:
    """
    ELBO loss for VampPrior VAE for colored MNIST.
    
    ELBO = E_q[log p(x|z)] - beta * KL(q(z|x) || p(z))
    
    Args:
        recon_x: Reconstructed input [batch_size, channels, height, width]
        x: Original input [batch_size, channels, height, width]
        mu: Posterior means [batch_size, latent_dim]
        log_var: Posterior log variances [batch_size, latent_dim]
        log_p_z: Log prior probability under VampPrior [batch_size]
        beta: Weight for KL term (for beta-VAE)
        recon_dist: Reconstruction distribution ("bce" or "mse")
    
    Returns:
        ELBO loss (to be minimized)
    """
    # Reconstruction loss (negative log likelihood)
    if recon_dist.lower() == "bce":
        # Binary cross-entropy works well for colored MNIST in [0,1] range
        assert 0 <= x.min() <= x.max() <= 1, "Input must be in [0,1] range for BCE loss"
        assert 0 <= recon_x.min() <= recon_x.max() <= 1, "Reconstruction must be in [0,1] range for BCE loss"
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif recon_dist.lower() == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:
        raise ValueError(f"Unknown recon_dist: {recon_dist}")
    
    # Average over batch
    recon_loss = recon_loss / x.shape[0]
    
    # KL divergence
    kl_loss_val = vamp_prior_kl_loss(mu, log_var, log_p_z)
    
    # Total ELBO loss (to be minimized)
    return recon_loss + beta * kl_loss_val