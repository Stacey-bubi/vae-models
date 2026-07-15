# vae-models

How three variational autoencoders — **VAE, IWAE, and VampPrior** — structure their latent spaces, and how that geometry affects sampling and interpolation. PyTorch, Colored MNIST. Group project for *Bayesian Methods of Machine Learning* (2025).

![Langevin latent refinement](notebooks/langevin.gif)

## Models

- **VAE** (Kingma & Welling, 2014) — standard Gaussian-prior baseline
- **IWAE** (Burda et al., 2016) — K-sample importance-weighted bound
- **VampPrior VAE** (Tomczak & Welling, 2018) — learned mixture-of-posteriors prior over pseudo-inputs

## What we measured

Generation quality with **FID** and **KID**, and latent geometry with a set of diagnostics: active units, per-dimension KL, PCA projections, a linear digit probe, latent traversals, and **MALA acceptance rates** along latent-space interpolations — a direct probe of posterior smoothness and connectivity.

## Results

**Sample quality vs. number of Langevin refinement steps** (lower is better):

| Model         | Best config | FID ↓     | KID ↓       |
| ------------- | ----------- | --------- | ----------- |
| VAE           | 20 steps    | 18.53     | 0.00890     |
| IWAE          | 5 steps     | 18.40     | 0.00869     |
| **VampPrior** | 20 steps    | **18.04** | **0.00810** |

- 5–20 refinement steps modestly improve samples; >=50 steps degrade them
- Linear-probe accuracy is near-ceiling for all models (VAE 0.994, IWAE 0.991, VampPrior 1.000), so the differences below come from **geometry**, not label separability:
  - **VAE** — one smooth, connected cloud; MALA acceptance stays high (>0.85) along interpolations
  - **IWAE** — fragmented; MALA acceptance drops to ~0 mid-interpolation, i.e. a sharp, poorly connected posterior
  - **VampPrior** — compact, separated clusters; high acceptance except at transitions between mixture components

Full tables and analysis in [`docs/results.md`](docs/results.md); the technical report PDF has the figures and is attached in docs/.

## Structure

```
vae_project/
  models/     VAE, IWAE, VampPrior, Langevin, latent analysis (MALA)
  train/      trainers, hooks, losses (ELBO, IWAE, VampPrior KL)
  dataset/    Colored MNIST loading and transforms
  evaluate.py FID / KID
notebooks/    experiments per model + comparison
docs/         results and analysis
```

## Setup

```bash
git clone https://github.com/Stacey-bubi/vae-models.git
cd vae-models

pixi install && pixi shell -e dev        # recommended
# or: python -m venv venv && source venv/bin/activate && pip install .
```

Then run the notebooks in `notebooks/` to reproduce the experiments.

## Authors

Viacheslav Chaunin, Alina Ermilova, Anastasiia Chernysheva - equal contribution.
Supervisor: Alexander Kolesov (Skoltech).

## License

MIT — see [LICENSE](LICENSE).
