from ..imports import *
from ..utils import *
from vae_project.models.langevin import langevin_dynamics
from vae_project.models.vae import VAE
from tqdm.auto import tqdm


class ReconstructDataset(Dataset):
    def __init__(self, dataloader, model: VAE, device=default_device, langevin_kwargs: dict = None):
        """
        Pre-computes reconstructions (optionally with Langevin dynamics) to cache them.
        Stores as uint8 to save memory and match evaluation metrics requirements.
        """
        self.imgs = []
        model.eval().to(device)
        
        for xb, *_ in tqdm(dataloader):
            xb = xb.to(device)
            
            if langevin_kwargs:
                _, recon, _ = langevin_dynamics(model, xb, **langevin_kwargs)
            else:
                with t.no_grad():
                    recon, *_ = model(xb)
            recon = to_uint8_img(recon).cpu()
            self.imgs.append(recon)

        self.imgs = t.cat(self.imgs)

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.imgs[i]