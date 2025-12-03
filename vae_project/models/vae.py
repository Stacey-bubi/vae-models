from torch.distributions import Normal
from ..imports import *
from ..utils import default_device, noop
from .blocks import Encoder, Decoder


class VAE(nn.Module):
    def __init__(
        self,
        encoder_or_channel_nums: Union[nn.Module, List[int]],
        decoder: Optional[nn.Module] = None,
        input_size: Optional[Tuple[int, int]] = None,
        latent_dim: int = 64,
        h_dim: int = 128,
        act_fn: nn.Module = nn.ReLU(),
        out_act: nn.Module = nn.Sigmoid(),
        recon_dist: Literal["bce", "mse"] = "bce",
        normalize: bool = False,
    ) -> None:
        """
        Base class for Variational Auto-Encoder model.

        Can be initialized in two ways:
        1. By passing pre-built `encoder` and `decoder` modules.
        2. By passing a list of `channel_nums` and `input_size` to build them internally.

        Args:
            encoder_or_channel_nums: An encoder module or a list of channel numbers.
            decoder: A decoder module (required if passing an encoder module).
            input_size (int): Image (height, width), required if building from channel numbers.
            latent_dim (int): Dimensionality of the latent space.
            out_act: Activation function after final layer
            recon_dist: Reconstruction distribution, either Normal or Bernoulli
            normalize (bool): Whether to normalize inputs to [-1,1]
        """
        super().__init__()
        self.out_act = out_act
        self.normalize, self.recon_dist = normalize, recon_dist

        if isinstance(encoder_or_channel_nums, nn.Module):
            self.encoder, self.decoder = encoder_or_channel_nums, decoder
        else:
            if input_size is None:
                raise ValueError("`input_size` must be provided when building from channel numbers")
            channels = encoder_or_channel_nums
            self.encoder = Encoder(channels, latent_dim=latent_dim, input_size=input_size, h_dim=h_dim, act_fn=act_fn)
            self.decoder = Decoder(
                channels[::-1],
                latent_dim=latent_dim,
                enc_final_shape=self.encoder.final_shape,
                h_dim=h_dim,
                act_fn=act_fn,
            )
        self._init_weights(act_fn)

    def _init_weights(self, act_fn):
        nonlin = act2str(act_fn)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean of the posterior q(z|x), shape [batch_size, latent_dim]
            log_var: Log variance of the posterior, shape [batch_size, latent_dim]

        Returns:
            z: Sampled latent codes of shape [batch_size, latent_dim]
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        if self.normalize:
            x = 2 * x - 1
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def decode(self, z: t.Tensor):
        return self.out_act(self.decoder(z))

    def sample(self, n: int = 1, device=default_device):
        self.eval()
        with t.no_grad():
            z = t.randn(n, self.encoder.latent_dim, device=device)
            return self.decode(z)

    @staticmethod
    def log_prior(z: t.Tensor):
        "Log probability of z under a standard Normal prior."
        return Normal(0, 1).log_prob(z).sum(dim=-1)

    def langevin_sample(self, n: int = 1, n_steps: int = 10, eps: float | tuple = 1e-3, recon_std=0.1, device=default_device):
        """
        Refines samples from the model's joint distribution p(x, z) using Langevin dynamics.

        Args:
            n (int): Number of samples to generate.
            n_steps (int): Number of MCMC refinement steps.
            eps (float or tuple): Step size for x and z. If float, same is used for both.
            recon_std (float): Std deviation of the reconstruction distribution p(x|z).
            device (str): The device to perform computation on.

        Returns:
            Tensor: A tensor of refined samples of shape [n, C, H, W].
        """
        self.eval()
        self.to(device)
        is_bce = self.recon_dist == "bce"

        eps_x, eps_z = (eps, eps) if isinstance(eps, (float, int)) else eps

        z = t.randn(n, self.decoder.latent_dim, device=device, requires_grad=True)
        with t.no_grad():
            x: t.Tensor = self.decode(z).detach()
        x.requires_grad = True

        for i in range(n_steps):
            # These are logits if BCE
            recon_mu = self.decoder(z)

            # log p(x|z)
            log_p_x_given_z = recon_log_prob(recon_mu, x, recon_std, self.recon_dist)
            log_p = log_p_x_given_z + self.log_prior(z).sum()

            self.zero_grad()
            log_p.backward()

            with t.no_grad():
                x.data += 0.5 * eps_x**2 * x.grad + eps_x * t.randn_like(x)
                z.data += 0.5 * eps_z**2 * z.grad + eps_z * t.randn_like(z)
                # make sure that target x is always in correct range
                if is_bce:
                    x.data.clamp_(0, 1)

            x.grad.zero_()
            z.grad.zero_()

        return x.detach()


def act2str(act_fn):
    """Returns string representation of activation function"""
    act_map = {nn.ReLU: "relu", nn.LeakyReLU: "leaky_relu"}
    return act_map.get(type(act_fn), "relu")


def recon_log_prob(recon_x, x, recon_std=0.1, recon_dist: Literal["bce", "mse"] = "bce"):
    """Calculates log p(x|z) based on the model's reconstruction distribution.
    This is the single source of truth for the reconstruction likelihood.
    """
    if recon_dist == "bce":
        return -F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
    if recon_dist == "mse":
        return -F.mse_loss(recon_x, x, reduction="sum") / (2 * recon_std**2)
    raise ValueError(f"Unknown recon_dist: '{self.recon_dist}'")
