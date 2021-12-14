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
        feature_vectors_list = x.unbind(1)
        # split vectors by features
        # process each separate vector
        out = [processor(feature) for processor, feature in zip(self.feature_processors, feature_vectors_list)]

        return out


class FeatureClassifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(FeatureClassifier, self).__init__()
        self.lin1 = nn.Linear(latent_dim, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.n_classes = n_classes

    def forward(self, x):
        x = self.lin1(x)
        x = self.sigmoid(x)
        return x

    def calculate_loss(self, x, target: torch.Tensor):
        criterion = nn.BCELoss()
        target -= 1
        # x.shape: (batch_size, n_classes)
        # target.shape: (batch_size, 1)
        target = torch.nn.functional.one_hot(target.long(), self.n_classes).float()

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
        loss = criterion(x.squeeze(1), target.float())
        return loss
