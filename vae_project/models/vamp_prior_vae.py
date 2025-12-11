from typing import Union, List, Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..imports import *
from ..utils import default_device
from .vae import VAE
from .blocks import Encoder, Decoder

class VampPrior(nn.Module):
    """
    VampPrior (Variational Mixture of Posteriors Prior) module for colored MNIST.
    
    This implements the core VampPrior mechanism from the paper:
    "VAE with a VampPrior" by Tomczak and Welling (2018).
    
    The prior is defined as a mixture of posteriors conditioned on learnable pseudo-inputs:
    p(z) = (1/K) * Σ_{k=1}^K q(z|u_k)
    
    For colored MNIST, we handle 3-channel RGB input instead of grayscale.
    """
    
    def __init__(self, 
                 num_components: int, 
                 input_channels: int, 
                 input_size: Tuple[int, int],
                 encoder: nn.Module,
                 use_pseudo_net: bool = False,
                 h_dim: int = 128,
                 act_fn: nn.Module = nn.ReLU()):
        """
        Initialize VampPrior module for colored MNIST.
        
        Args:
            num_components: Number of pseudo-inputs (mixture components)
            input_channels: Number of input channels (3 for RGB colored MNIST)
            input_size: Input image dimensions (height, width)
            encoder: Encoder network that maps inputs to (mu, log_var)
            use_pseudo_net: Whether to use network to generate pseudo-inputs (True) or direct parameters (False)
            h_dim: Hidden dimension size for pseudo-input network (if used)
            act_fn: Activation function for pseudo-input network
        """
        super().__init__()
        self.num_components = num_components
        self.input_channels = input_channels
        self.input_size = input_size
        self.input_height, self.input_width = input_size
        self.encoder = encoder
        self.use_pseudo_net = use_pseudo_net
        
        # For colored MNIST, pseudo-inputs should be in [0,1] range for RGB
        if use_pseudo_net:
            # Network approach: generate pseudo-inputs from identity matrix
            total_pixels = input_channels * self.input_height * self.input_width
            self.pseudo_net = nn.Sequential(
                nn.Linear(num_components, h_dim),
                act_fn,
                nn.Linear(h_dim, h_dim),
                act_fn,
                nn.Linear(h_dim, total_pixels),
                nn.Sigmoid()  # Better than Hardtanh for gradient flow
            )
            # Identity matrix input for generating pseudo-inputs (device will be handled automatically)
            self.register_buffer('idle_input', torch.eye(num_components, num_components))
        else:
            # Direct parameter approach: learnable pseudo-inputs tensor
            self.pseudoinputs = nn.Parameter(
                torch.empty(num_components, input_channels, self.input_height, self.input_width)
            )
            self._init_pseudoinputs()

    def _init_pseudoinputs(self):
        """Initialize pseudo-inputs to be near colored MNIST data manifold."""
        # Initialize with slight color variations around gray (0.5)
        nn.init.constant_(self.pseudoinputs, 0.5)
        
        # Add channel-specific noise - slightly more variation in color channels
        device = self.pseudoinputs.device
        for c in range(self.input_channels):
            noise = torch.randn(self.num_components, 1, self.input_height, self.input_width, device=device) * 0.1
            if c > 0:  # Add more variation to color channels (not just grayscale)
                noise = noise * 1.5
            self.pseudoinputs.data[:, c:c+1] += noise
        
        # Clamp to valid [0,1] range for RGB
        self.pseudoinputs.data = torch.clamp(self.pseudoinputs.data, 0.0, 1.0)

    def get_device(self):
        """Get the device of the module parameters."""
        return next(self.parameters()).device

    def get_pseudo_inputs(self) -> torch.Tensor:
        """
        Get all pseudo-inputs.
        
        Returns:
            pseudo_inputs: Tensor of shape [num_components, channels, height, width]
        """
        device = self.get_device()
        
        if self.use_pseudo_net:
            # Generate pseudo-inputs through network
            pseudo_flat = self.pseudo_net(self.idle_input.to(device))  # [K, C*H*W]
            pseudo_inputs = pseudo_flat.view(
                -1, 
                self.input_channels, 
                self.input_height, 
                self.input_width
            )
        else:
            # Use direct learnable parameters
            pseudo_inputs = self.pseudoinputs.to(device)
        
        return pseudo_inputs

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior parameters for all pseudo-inputs.
        
        Returns:
            mu_p: Means of posteriors for pseudo-inputs, shape [num_components, latent_dim]
            log_var_p: Log variances of posteriors for pseudo-inputs, shape [num_components, latent_dim]
        """
        pseudo_inputs = self.get_pseudo_inputs()
        # Get posterior parameters for each pseudo-input
        mu_p, log_var_p = self.encoder(pseudo_inputs)
        return mu_p.to(self.get_device()), log_var_p.to(self.get_device())
    
    @torch.no_grad()
    def _compute_component_log_probs(
        self, 
        z: torch.Tensor, 
        mu_p_batch: torch.Tensor, 
        log_var_p_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities under Gaussian components for a batch of pseudo-inputs.
        
        Args:
            z: Latent samples [batch_size, latent_dim]
            mu_p_batch: Means for batch of components [batch_size_components, latent_dim]
            log_var_p_batch: Log variances for batch of components [batch_size_components, latent_dim]
            
        Returns:
            Log probabilities [batch_size, batch_size_components]
        """
        device = z.device
        latent_dim = z.shape[-1]
        z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        mu_expanded = mu_p_batch.unsqueeze(0).to(device)  # [1, batch_size_components, latent_dim]
        log_var_expanded = log_var_p_batch.unsqueeze(0).to(device)  # [1, batch_size_components, latent_dim]
        
        # Compute log probability under each Gaussian component
        log_prob = -0.5 * (
            log_var_expanded + 
            (z_expanded - mu_expanded) ** 2 / torch.exp(log_var_expanded) +
            torch.log(2 * torch.tensor(np.pi, device=device))
        ).sum(-1)  # Sum over latent dimensions -> [batch_size, batch_size_components]
        
        return log_prob

    def log_p_z(self, z: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        """
        Compute log p(z) under VampPrior for given z samples.
        
        This uses batched processing for memory efficiency and log-sum-exp for numerical stability.
        
        Args:
            z: Latent samples, shape [batch_size, latent_dim]
            batch_size: Batch size for processing pseudo-inputs (memory optimization)
        
        Returns:
            log_p_z: Log prior probability, shape [batch_size]
        """
        num_components = self.num_components
        device = z.device
        
        # Pre-allocate tensor for log probabilities
        log_probs_all = torch.empty(z.shape[0], num_components, device=device)
        
        # Process in smaller batches for memory stability
        for i in range(0, num_components, batch_size):
            end = min(i + batch_size, num_components)
            indices = torch.arange(i, end, device=device)
            mu_p_batch, log_var_p_batch = self._get_posteriors_for_indices(indices)
            
            # More stable Gaussian log-likelihood calculation
            z_expanded = z.unsqueeze(1)  # [B, 1, D]
            mu_expanded = mu_p_batch.unsqueeze(0)  # [1, K_batch, D]
            log_var_expanded = log_var_p_batch.unsqueeze(0)  # [1, K_batch, D]
            
            # Prevent numerical overflow in exp(log_var)
            log_var_clamped = torch.clamp(log_var_expanded, -10, 10)
            var = torch.exp(log_var_clamped)
            
            # Compute log probability with numerical stability
            log_prob = -0.5 * (
                torch.log(2 * torch.tensor(np.pi, device=device)) + 
                log_var_clamped + 
                (z_expanded - mu_expanded) ** 2 / (var + 1e-8)
            ).sum(-1)  # [B, K_batch]
            
            log_probs_all[:, i:end] = log_prob
        
        # Stable log-sum-exp for mixture
        log_weights = -torch.log(torch.tensor(num_components, dtype=torch.float32, device=device))
        log_component_probs = log_probs_all + log_weights
        
        # Proper log-sum-exp for numerical stability
        max_log = torch.max(log_component_probs, dim=1, keepdim=True)[0]
        log_p_z = max_log.squeeze(1) + torch.logsumexp(log_component_probs - max_log, dim=1)
        
        return log_p_z
    
    def _get_posteriors_for_indices(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior parameters for specific pseudo-input indices."""
        pseudo_inputs = self.get_pseudo_inputs()[indices]
        mu_p, log_var_p = self.encoder(pseudo_inputs)
        return mu_p.to(self.get_device()), log_var_p.to(self.get_device())


class VampPriorVAE(VAE):
    """
    Variational Autoencoder with VampPrior for colored MNIST.
    
    This implements a VAE where the standard Gaussian prior is replaced with a
    VampPrior (mixture of posteriors from learnable pseudo-inputs).
    
    Designed specifically for colored MNIST with:
    - Input range [0,1] for RGB channels
    - 3 input channels for RGB
    - Binary cross-entropy reconstruction loss
    """
    
    def __init__(
        self,
        encoder_or_channel_nums: Union[nn.Module, List[int]],
        decoder: Optional[nn.Module] = None,
        input_size: Optional[Tuple[int, int]] = None,
        input_channels: int = 3,
        latent_dim: int = 64,
        num_components: int = 500,
        h_dim: int = 256,
        act_fn: nn.Module = nn.ReLU(),
        out_act: nn.Module = nn.Sigmoid(),
        normalize: bool = False,
        use_pseudo_net: bool = False,
        **kwargs  # Accept and ignore any extra kwargs like 'device'
    ):
        """
        Initialize VampPrior VAE for colored MNIST.
        
        Args:
            input_size: Image dimensions (height, width), for MNIST: (28, 28)
            input_channels: Number of input channels (3 for RGB colored MNIST)
            num_components: Number of pseudo-inputs, typically 500
            use_pseudo_net: False for direct parameters (faster), True for network generation
        """
        super().__init__(
            encoder_or_channel_nums=encoder_or_channel_nums,
            decoder=decoder,
            input_size=input_size,
            latent_dim=latent_dim,
            h_dim=h_dim,
            act_fn=act_fn,
            out_act=out_act,
            recon_dist="bce",
            normalize=normalize
        )
        
        if input_size is None:
            raise ValueError("input_size must be provided for VampPriorVAE")
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_components = num_components
        
        # Initialize VampPrior (device will be handled automatically when model is moved)
        self.prior = VampPrior(
            num_components=num_components,
            input_channels=input_channels,
            input_size=input_size,
            encoder=self.encoder,
            use_pseudo_net=use_pseudo_net,
            h_dim=h_dim,
            act_fn=act_fn
        )
        
        # Store z for VampPrior computation
        self.z = None
        
        # Validate pseudo-inputs after initialization
        self._validate_initialization()

    def _validate_initialization(self):
        """Validate that the model is properly initialized for colored MNIST."""
        with torch.no_grad():
            device = next(self.parameters()).device
            # Check pseudo-inputs are in valid range
            pseudo_inputs = self.prior.get_pseudo_inputs()
            assert 0.0 <= pseudo_inputs.min() <= pseudo_inputs.max() <= 1.0, \
                f"Pseudo-inputs out of [0,1] range: min={pseudo_inputs.min():.4f}, max={pseudo_inputs.max():.4f}"
            
            # Check channel dimensions
            assert pseudo_inputs.shape[1] == self.input_channels, \
                f"Expected {self.input_channels} channels, got {pseudo_inputs.shape[1]}"
            
            # Check we can compute forward pass
            test_input = torch.zeros(1, self.input_channels, *self.input_size, device=device)
            try:
                recon_x, mu, log_var = self(test_input)
                print("✓ Model initialization validated successfully for colored MNIST")
            except Exception as e:
                print(f"⚠ Model initialization warning: {str(e)}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VampPrior VAE for colored MNIST.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple containing:
            - recon_x: Reconstructed input [batch_size, channels, height, width]
            - mu: Posterior means [batch_size, latent_dim]
            - log_var: Posterior log variances [batch_size, latent_dim]
        """
        # Store z for VampPrior computation
        recon_x, mu, log_var = super().forward(x)
        self.z = self.reparameterize(mu, log_var)  # Store for trainer access
        return recon_x, mu, log_var

    def get_device(self):
        """Get the device of the module parameters."""
        return next(self.parameters()).device

    def sample(self, n: int = 1, device=None) -> torch.Tensor:
        """
        Generate samples from the VampPrior for colored MNIST.
        
        This generates new colored digit-like images.
        
        Args:
            n: Number of samples to generate
            device: Device to generate samples on (if None, uses model's device)
            
        Returns:
            Generated samples [n, channels, height, width]
        """
        if device is None:
            device = self.get_device()
            
        self.eval()
        with torch.no_grad():
            # Sample component indices uniformly
            component_indices = torch.randint(0, self.num_components, (n,), device=device)
            
            # Get posterior parameters for selected components
            mu_p, log_var_p = self.prior._get_posteriors_for_indices(component_indices)
            
            # Sample z from selected components
            z = self.reparameterize(mu_p, log_var_p)
            
            # Decode to generate samples
            return self.decode(z)

    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """More numerically stable reparameterization"""
        std = t.exp(0.5 * t.clamp(log_var, -10, 10))  # Prevent overflow
        eps = t.randn_like(std)
        return mu + eps * std
    
    def log_prior(self, z: t.Tensor):
        """Override base VAE prior - not used but prevents bugs"""
        return self.prior.log_p_z(z)