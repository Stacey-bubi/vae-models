# latent analysis tools

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch, math, numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_latent_codes(model, dataloader, device):
    model.eval()
    all_mu, all_logvar, all_z, all_labels = [], [], [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)

            mu, logvar = model.encoder(xb)
            z = model.reparameterize(mu, logvar)

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_z.append(z.cpu())
            all_labels.append(yb.cpu())

    return (
        torch.cat(all_mu),
        torch.cat(all_logvar),
        torch.cat(all_z),
        torch.cat(all_labels),
    )


def compute_active_units(z, threshold=0.01):
    var = z.var(dim=0)
    active = (var > threshold).sum().item()
    return var, active


def kl_from_mu_logvar(mu, logvar):
    """
    KL(q(z|x) || p(z)) for each dimension.
    Input:  mu, logvar — tensors of shape (N, D)
    Output: (D,)
    """
    kl = 0.5 * (mu**2 + torch.exp(logvar) - logvar - 1)
    return kl.mean(dim=0)




def linear_probe(z, y):
    clf = LogisticRegression(max_iter=200)
    clf.fit(z, y)

    acc = clf.score(z, y)
    return clf, acc


def plot_latent_2d(z, y, method="pca", title=""):
    z = z.numpy()
    y = y.numpy()

    if method == "pca":
        z2 = PCA(n_components=2).fit_transform(z)
    elif method == "tsne":
        z2 = TSNE(n_components=2, perplexity=30, learning_rate="auto").fit_transform(z)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(6, 6))
    plt.scatter(z2[:, 0], z2[:, 1], c=y, s=3, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def latent_traversal(model, x, dim, steps=8, sigma_scale=3, device="cpu"):
    model.eval()

    with torch.no_grad():
        mu, logvar = model.encoder(x.to(device))
        mu = mu.squeeze()
        logvar = logvar.squeeze()

        z0 = mu.clone()
        sigma = (0.5 * logvar[dim]).exp().item()

    values = torch.linspace(
        z0[dim] - sigma_scale * sigma,
        z0[dim] + sigma_scale * sigma,
        steps
    )

    imgs = []
    with torch.no_grad():
        for v in values:
            z = z0.clone()
            z[dim] = v
            img = model.decoder(z.unsqueeze(0)).cpu()
            imgs.append(img)

    return torch.cat(imgs, dim=0)


def log_px_given_z(model, z, x):
    recon = model.decode(z)                    # [B,C,H,W]
    logp = -F.binary_cross_entropy(recon, x.expand(recon.shape), reduction='none')
    return logp.view(logp.shape[0], -1).sum(dim=1)  # [B]

def log_prior(model, z):
    if hasattr(model, "log_prior"):
        return model.log_prior(z)
    else:
        return torch.distributions.Normal(0,1).log_prob(z).sum(dim=1)

def log_target(model, z, x):
    return log_px_given_z(model, z, x) + log_prior(model, z)

# MALA relax: short chain starting at z_init (1,D) returning final z
def mala_relax(model, x, z_init, n_steps=100, eps=0.05, device="cpu"):
    model.to(device)
    x = x.to(device)
    z = z_init.clone().to(device)
    if z.dim()==1: z = z.unsqueeze(0)
    D = z.shape[-1]
    current = z.clone()
    current_logp = log_target(model, current, x).detach()
    accepts = 0
    for t in range(n_steps):
        current.requires_grad_(True)
        logp = log_target(model, current, x).sum()
        grad = torch.autograd.grad(logp, current)[0]  # [1,D]
        proposal_mean = current + 0.5 * (eps**2) * grad
        noise = torch.randn_like(current)
        proposed = proposal_mean + eps * noise

        proposed_logp = log_target(model, proposed, x).detach()
        # reverse grad
        proposed.requires_grad_(True)
        rev_logp = log_target(model, proposed, x).sum()
        proposed_grad = torch.autograd.grad(rev_logp, proposed)[0]
        # compute log q forward/reverse (Gaussian)
        var = eps**2
        log_q_forward = -0.5 * (((proposed - proposal_mean)**2) / var).sum(dim=1) - 0.5*D*math.log(2*math.pi*var)
        reverse_mean = proposed + 0.5 * (eps**2) * proposed_grad
        log_q_reverse = -0.5 * (((current - reverse_mean)**2) / var).sum(dim=1) - 0.5*D*math.log(2*math.pi*var)

        log_alpha = proposed_logp + log_q_reverse - current_logp - log_q_forward
        accept = (torch.log(torch.rand(1, device=device)) < log_alpha).item()
        if accept:
            current = proposed.detach()
            current_logp = proposed_logp
            accepts += 1
        # else keep current
    acc_rate = accepts / n_steps
    return current.squeeze(0).cpu(), acc_rate

# main function: MCMC-relaxed interpolation between z_a and z_b 
def mcmc_relaxed_interpolation(model, x_a, x_b, z_a, z_b, n_t=9, n_relax=100, eps=0.05, device="cpu"):
    ts = np.linspace(0,1,n_t)
    z_lin = [(1-t)*z_a + t*z_b for t in ts]
    z_relaxed = []
    accs = []
    # We'll evaluate logp wrt BOTH x_a and x_b for diagnostics
    for z0 in z_lin:
        z0 = z0.to(device)
        z_rel, acc = mala_relax(model, x_a.unsqueeze(0), z0, n_steps=n_relax, eps=eps, device=device)
        z_relaxed.append(z_rel)
        accs.append(acc)
    z_lin = torch.stack([z.cpu() if isinstance(z, torch.Tensor) else torch.tensor(z).cpu() for z in z_lin], dim=0)
    z_relaxed = torch.stack(z_relaxed, dim=0)
    # compute log-targets
    with torch.no_grad():
        logp_lin_a = log_target(model, z_lin.to(device), x_a.unsqueeze(0).to(device)).cpu().numpy()
        logp_rel_a = log_target(model, z_relaxed.to(device), x_a.unsqueeze(0).to(device)).cpu().numpy()
        logp_lin_b = log_target(model, z_lin.to(device), x_b.unsqueeze(0).to(device)).cpu().numpy()
        logp_rel_b = log_target(model, z_relaxed.to(device), x_b.unsqueeze(0).to(device)).cpu().numpy()
    return ts, z_lin, z_relaxed, (logp_lin_a, logp_rel_a, logp_lin_b, logp_rel_b), accs

def decode_grid(zs, model, nrow=9):
    with torch.no_grad():
        imgs = model.decode(zs.to("cpu")).cpu()
    grid = make_grid(imgs, nrow=nrow, normalize=False)
    plt.figure(figsize=(9,3))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.show()
