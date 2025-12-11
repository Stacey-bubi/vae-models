from ..imports import *


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
        self.channel_nums, self.act_fn, self.latent_dim, self.h_dim = channel_nums, act_fn, latent_dim, h_dim

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
    def __init__(self, channels_n, latent_dim, enc_final_shape, h_dim=128, act_fn=nn.ReLU()):
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
                    act_fn,
                ]
            )
        self.deconvs = nn.Sequential(*deconvs[:-1])

    def forward(self, z):
        x = self.act_fn(self.fc1(z))
        x = self.act_fn(self.fc2(x))
        x = x.view(-1, self.start_c, self.start_h, self.start_w)
        return self.deconvs(x)


class SpectralNormDecoder(Decoder):
    def __init__(self, channels_n, latent_dim, enc_final_shape, h_dim=128, act_fn=nn.ReLU()):
        super().__init__(channels_n, latent_dim, enc_final_shape, h_dim, act_fn)

        deconvs = []
        for layer in self.deconvs:
            if isinstance(layer, (nn.ConvTranspose2d, nn.Linear)):
                layer = nn.utils.spectral_norm(layer)
            deconvs.append(layer)
        self.deconvs = nn.Sequential(*deconvs)
