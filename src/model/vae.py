from typing import Tuple, List, Union

import torch
from torch import nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.feature import FeatureClassifier, FeatureRegressor, FeatureLayer
from src.model.feedforward import FeedForward


class VAE(nn.Module):
    def __init__(self, feature_processors: List[Union[FeatureClassifier, FeatureRegressor]],
                 image_size: Tuple[int, int, int] = (1, 64, 64),
                 latent_dim: int = 1024):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.num_pixels = self.image_size[1] * self.image_size[2]
        self.encoder = Encoder(image_size, latent_dim)
        self.decoder = Decoder(image_size, latent_dim)

        self.feature_layer = FeatureLayer(latent_dim=latent_dim,
                                          feature_processors=feature_processors)
        self.feed_forward = FeedForward(latent_dim=latent_dim)

        self.n_features = len(feature_processors)

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
            eps = torch.randn_like(std)
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
        latent_sample = torch.unsqueeze(latent_sample, 0)
        z_i = self.feed_forward(latent_sample)
        return z_i

    def decode_latent(self, x, reduce=True):
        if reduce:
            x = torch.mean(x, 1)
        return self.decoder(x)

    def latent_operations(self, x1, x2, properties: List[bool]):
        sample_x1 = self.sample_latent(x1)
        sample_x2 = self.sample_latent(x2)

        latent_out = torch.zeros_like(sample_x1)

        for i in range(len(properties)):
            latent_out[0][i * self.latent_dim:(i + 1) * self.latent_dim] = sample_x1[0][i * self.latent_dim: (i + 1) * self.latent_dim] if  properties[i] else sample_x2[0][i * self.latent_dim: (i + 1) * self.latent_dim]

        out = self.decode_latent(latent_out)

        return out

    def forward(self, x):
        mean, log_var = self.encoder(x)

        latent_sample = self.reparameterize(mean, log_var)

        # x.shape = (batch_size, n_features * latent_dim)
        # making shape = (batch_size, n_features, latent_dim)
        z_i = latent_sample.view(-1, self.n_features, self.latent_dim)

        feature_discr = self.feature_layer(z_i)

        # (batch_size, n_features, latent_dim)
        # sum by features
        z_i = self.feed_forward(z_i)

        scene = sum(z_i)
        reconstruct = self.decoder(scene)

        return reconstruct, mean, log_var, latent_sample, feature_discr
