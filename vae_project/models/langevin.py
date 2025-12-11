from ..imports import *
from ..utils import *
from ..models import VAE
from IPython.display import display, Image as IPImage
from PIL import Image, ImageDraw


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


from torchvision.utils import make_grid

def animate_langevin(latent_seq: t.Tensor, model: VAE, fn="langevin.gif", fps=20, scale=4):
    """Generates a grid animation from a batch of latent histories [B, T, Z]"""
    if latent_seq.ndim == 2: latent_seq = latent_seq.unsqueeze(0)
    B, T, Z = latent_seq.shape
    dev = get_model_device(model)
    
    frames = []
    for t_step in range(T):
        z = latent_seq[:, t_step].to(dev)
        with t.no_grad():
            batch_imgs = model.decode(z)
        
        if batch_imgs.min() < 0: batch_imgs = (batch_imgs + 1) / 2
        
        # Create grid for this timestamp: [C, H_grid, W_grid]
        grid = make_grid(batch_imgs.clamp(0,1), nrow=int(B**0.5), padding=2, pad_value=1)
        frames.append(grid.cpu())

    # Stack to [T, C, H, W] and upscale
    vid = t.stack(frames)
    if scale != 1:
        vid = F.interpolate(vid, scale_factor=scale, mode="nearest")
    
    # Convert to [T, H, W, C] uint8 for PIL
    vid = to_uint8_img(vid).permute(0, 2, 3, 1).numpy()
    
    pil_imgs = []
    for i, frame in enumerate(vid):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Step: {i}", fill=(255, 0, 0))
        pil_imgs.append(img)

    pil_imgs[0].save(fn, save_all=True, append_images=pil_imgs[1:], duration=1000 // fps, loop=0)
    print(f"Saved animation to {fn}")
    return IPImage(filename=fn)