
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_latent_codes(model, loader, device="cpu"):
    model.eval()
    zs = []
    ys = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, logvar = model.encoder.forward_to_mu_logvar(x)

            # VAE/Vamp: (batch, latent_dim)
            # IWAE: (batch, 1, latent_dim) → squeeze
            mu = mu.squeeze()

            zs.append(mu.cpu())
            ys.append(y)

    return torch.cat(zs), torch.cat(ys)


def compute_active_units(z, threshold=0.01):
    var = z.var(dim=0)
    active = (var > threshold).sum().item()
    return var, active


def kl_per_dimension(model, loader, device="cpu"):
    model.eval()
    all_kl = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            # Универсальный вызов (VAE / IWAE / VampPrior)
            mu, logvar = model.encoder.forward_to_mu_logvar(x)

            # IWAE может вернуть shape (B,1,D)
            mu = mu.squeeze()
            logvar = logvar.squeeze()

            kl = 0.5 * (mu**2 + torch.exp(logvar) - logvar - 1)

            all_kl.append(kl.mean(0).cpu())

    return torch.stack(all_kl).mean(0)



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
        mu, logvar = model.encoder.forward_to_mu_logvar(x.to(device))
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


