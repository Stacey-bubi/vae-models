from ..imports import *
from .vae import VAE


class IWAE(VAE):
    """Importance-Weighted Autoencoder.
    Uses importance weighting during training and evaluation
    to compute a tighter lower bound on the log-likelihood.
    """

    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor, K: int = 1) -> t.Tensor:
        """Reparameterization trick with K importance samples.
        
        Args:
            mu: Mean of the posterior q(z|x), shape [batch_size, latent_dim]
            log_var: Log variance of the posterior, shape [batch_size, latent_dim]
            K: Number of importance samples to draw
            
        Returns:
            z: Sampled latent codes of shape [batch_size * K, latent_dim]
        """
        batch_size = mu.shape[0]
        std = t.exp(0.5 * log_var)
        eps = t.randn(batch_size, K, mu.shape[1], device=mu.device)
        z = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        return z
    
    def forward(self, x: t.Tensor, K: int = 1) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        """Forward pass that returns K samples for importance weighting.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            K: Number of importance samples
            
        Returns:
            reconstructions: [batch_size, K, channels, height, width]
            z: [batch_size, K, latent_dim] - the sampled latent codes
            mu: [batch_size, latent_dim]
            log_var: [batch_size, latent_dim]
        """
        bs = x.shape[0]
        if self.normalize:
            x = 2 * x - 1
        mu, log_var = self.encoder(x)
        
        z = self.reparameterize(mu, log_var, K)
        recon = self.decoder(z.view(bs * K, -1))

        return recon.view(bs, K, *recon.shape[1:]), z, mu, log_var
