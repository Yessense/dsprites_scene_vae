import torch

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, latent_dim):
        super(FeedForward, self).__init__()
        self.latent_dim = latent_dim

        # Activation function
        self.activation = nn.GELU()

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, latent_dim * 2)
        self.lin2 = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        feature_vectors_list = x.unbind(1)

        out = []

        for feature in feature_vectors_list:
            feature = self.lin1(feature)
            feature = self.activation(feature)
            feature = self.lin2(feature)
            feature = self.activation(feature)

            out.append(feature)

        return out







