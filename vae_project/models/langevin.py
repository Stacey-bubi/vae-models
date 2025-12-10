from ..imports import *
from ..models import VAE


def langevin_dynamics(model: VAE, x: t.Tensor, n_steps=100, step_sz=1e-2, noise_scale=1.0, return_hist=False):
    """
    Refines latent z using Langevin dynamics on prior p(z).

    Args:
        return_hist (bool): If True, returns list of latent vectors at each step.

    Returns:
        (x_orig, x_ref, z_ref, hist) if return_hist=True
        (x_orig, x_ref, z_ref)       otherwise
    """
    model.eval()

    with t.no_grad():
        if model.normalize:
            x = 2 * x - 1
        mu, log_var = model.encoder(x)

        z = model.reparameterize(mu, log_var)
        if z.ndim == 3:
            z = z.squeeze(1)

        x_orig = model.decode(z)

    z = z.detach().clone().requires_grad_(True)
    hist = [z.detach().cpu()] if return_hist else None

    for _ in range(n_steps):
        log_p = model.log_prior(z).sum()
        grad_z = t.autograd.grad(log_p, z)[0]

        noise = t.randn_like(z) * noise_scale
        z.data = z.data + 0.5 * step_sz * grad_z + (step_sz**0.5) * noise

        if return_hist:
            hist.append(z.detach().cpu())

    with t.no_grad():
        x_ref = model.decode(z)

    return (x_orig, x_ref, z.detach(), t.stack(hist, dim=1)) if return_hist else (x_orig, x_ref, z.detach())
