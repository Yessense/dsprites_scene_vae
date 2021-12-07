from typing import Tuple

import torch
from torch import nn

from src.model.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (1, 32, 32), latent_dim: int = 6):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_pixels = self.image_size[1] * self.image_size[2]
        self.encoder = Encoder(image_size, latent_dim)
        # self.decoder =

    def reparameterize(self, mean, logvar):
        """

        Parameters
        ----------
        mu: torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar: torch.Tensor
            Diagonal log variance of the normal distribution.
            Shape (batch_size, latent_dim)

        """

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std)
            return mean + std * eps
        else:
            return mean

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x: torch.Tensor
            Batch of data. Shape (batch_size, n_channels, height, width)
        """

        mean, log_var = self.encoder(x)
        latent_sample = self.reparameterize(mean, log_var)
        return latent_sample

    def forward(self, x):
        mean, log_var = self.encoder(x)
        latent_sample = self.reparameterize(mean, log_var)
        reconstruct = self.decoder(latent_sample)

        return reconstruct, mean, log_var, latent_sample
