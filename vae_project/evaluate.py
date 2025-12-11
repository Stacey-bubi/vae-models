import torch_fidelity
from .models import VAE
from .imports import *
from .utils import default_device


class Generator(nn.Module):
    """Wrapper to match `torch_fidelity` api. Must return batch of uint8 images"""

    def __init__(self, model: VAE):
        super().__init__()
        self.model = model

    def forward(self, z):
        images = self.model.decode(z)
        return (images * 255).to(t.uint8)


def evaluate_model(
    model: VAE, dataset, n_samples: int = 2_000, fid=True, isc=True, kid=True, verbose=True, device=default_device
):
    """Evaluates generation quality of a model"""
    gen = Generator(model)
    wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(
        gen, z_size=model.encoder.latent_dim, z_type="normal", num_classes=0
    )

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=wrapped_generator,
        input1_model_num_samples=n_samples,
        input2=dataset,
        cuda=True if device.type == "cuda" else False,
        isc=isc,
        fid=fid,
        kid=kid,
        verbose=verbose,
    )
    return metrics_dict


def evaluate_reconstructions(
    target_dataset: Dataset, recon_dataset: Dataset, fid=True, isc=False, kid=True, verbose=True, device=default_device
):
    '''Evaluates quality of reconstructions'''
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=target_dataset,
        input2=recon_dataset,
        cuda=True if device.type == "cuda" else False,
        isc=isc,
        fid=fid,
        kid=kid,
        verbose=verbose,
    )
    return metrics_dict
