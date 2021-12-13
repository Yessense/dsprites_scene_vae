from typing import List

import numpy as np
import torch
from torch import nn


class FeatureLayer(nn.Module):
    def __init__(self, latent_dim, feature_processors: List[nn.Module]):
        super(FeatureLayer, self).__init__()
        self.latent_dim = latent_dim
        self.n_features = len(feature_processors)
        self.feature_processors: List[nn.Module] = feature_processors

    def forward(self, x):
        # x.shape = (batch_size, n_features * latent_dim)
        # making shape = (batch_size, n_features, latent_dim)
        feature_vectors_matrix = x.view(-1, self.n_features, self.latent_dim)

        # split vectors by features
        feature_vectors_list = x.unbind(1)

        # process each separate vector
        out = [processor(x) for processor, feature in zip(self.feature_processors, feature_vectors_list)]

        return out


class FeatureClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(FeatureClassifier, self).__init__()
        self.lin1 = nn.Linear(latent_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = self.lin1(x)
        return x

    def calculate_loss(self, x, target):
        criterion = nn.BCELoss()

        # x.shape: (batch_size, n_classes)
        # target.shape: (batch_size, 1)
        target = torch.nn.functional.one_hot(target, self.n_classes)

        loss = criterion(x, target)
        return loss





class FeatureRegressor(nn.Module):
    def __init__(self, latent_dim):
        super(FeatureRegressor, self).__init__()
        self.lin1 = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x = self.lin1(x)
        return x

    def calculate_loss(self, x, target):
        criterion = nn.MSELoss()
        loss = criterion(x, target)
        return loss

