from ..imports import *
from ..utils import default_device


class Encoder(nn.Module):
    def __init__(
        self,
        channel_nums: list[int],
        latent_dim: int,
        input_size: Tuple[int, int],
        h_dim: int = 128,
        act_fn: nn.Module = nn.ReLU(),
    ) -> None:
        """
        Convolutional Encoder for VAE.

        Args:
            channel_nums: List of channels for conv layers, e.g., [1, 16, 32, 64].
            latent_dim: Dimensionality of the latent space.
            input_size: The (height, width) of the input images.
        """
        super().__init__()
        self.act_fn, self.latent_dim, self.h_dim = act_fn, latent_dim, h_dim

        convs = []
        for i in range(len(channel_nums) - 1):
            convs.extend([nn.Conv2d(channel_nums[i], channel_nums[i + 1], kernel_size=3, stride=2, padding=1), act_fn])
        self.convs = nn.Sequential(*convs)

        with t.no_grad():
            dummy_input = t.zeros(1, channel_nums[0], *input_size)
            self.final_shape = self.convs(dummy_input).shape
            self.flat_sz = self.convs(dummy_input).view(-1).shape[0]

        self.fc1 = nn.Linear(self.flat_sz, h_dim)
        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_log_var = nn.Linear(h_dim, latent_dim)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        x = self.convs(x)
        x = x.view(-1, self.flat_sz)
        x = self.act_fn(self.fc1(x))
        return self.fc_mu(x), self.fc_log_var(x)


class Decoder(nn.Module):
    def __init__(self, channels_n, latent_dim, enc_final_shape, h_dim=128, act_fn=nn.ReLU(), out_act=nn.Sigmoid()):
        """
        Convolutional Decoder for VAE.

        Args:
            channels_n: List of channels for deconv layers, typically reversed from encoder.
            latent_dim: Dimensionality of the latent space.
            enc_final_shape: The output shape of the encoder's conv layers.
        """
        super().__init__()
        self.act_fn, self.latent_dim, self.h_dim = act_fn, latent_dim, h_dim

        self.start_c, self.start_h, self.start_w = enc_final_shape[1:]
        self.fc_flat_sz = self.start_c * self.start_h * self.start_w

        self.fc1 = nn.Linear(latent_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, self.fc_flat_sz)

        deconvs = []
        for i in range(len(channels_n) - 1):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(channels_n[i], channels_n[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    act_fn if i < len(channels_n) - 2 else out_act,
                ]
            )
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, z):
        x = self.act_fn(self.fc1(z))
        x = self.act_fn(self.fc2(x))
        x = x.view(-1, self.start_c, self.start_h, self.start_w)
        return self.deconvs(x)


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
        normalize: bool = False
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
            normalize (bool): Whether to normalize inputs to [-1,1]
        """
        super().__init__()
        self.normalize = normalize
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
                out_act=out_act
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
        if self.normalize: x = 2*x-1
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample(self, n, device=default_device):
        self.eval()
        with t.no_grad():
            z = t.randn(n, self.encoder.latent_dim).to(device)
            return self.decoder(z)


def act2str(act_fn):
    """Returns string representation of activation function"""
    act_map = {nn.ReLU: "relu", nn.LeakyReLU: "leaky_relu"}
    return act_map.get(type(act_fn), "relu")
