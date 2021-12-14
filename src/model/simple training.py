from itertools import product
from random import random, choices
from typing import List, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt
import logging

from src.model.feature import FeatureClassifier, FeatureRegressor

logging.basicConfig(level=logging.INFO)
from src.model.vae import VAE
from src.utils.dataset import get_dataloader

np.set_printoptions(suppress=True)


def kld_loss(mean, log_var):
    loss = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss


def bce_loss(x, reconstruction):
    loss = torch.nn.BCELoss(reduction='sum')
    return loss(reconstruction, x)


def vae_loss(x, reconstruction,
             mean, log_var,
             y_preds, y_trues, feature_processors):
    bce = bce_loss(x, reconstruction)
    feat = feature_loss(y_preds, y_trues, feature_processors)
    kld = kld_loss(mean, log_var)

    return bce + kld +  100 * feat, bce, kld, feat


def feature_loss(y_preds, y_trues, feature_processors: List[Union[FeatureClassifier, FeatureRegressor]]):
    losses = [processor.calculate_loss(y_pred, y_true) for y_pred, y_true, processor in
              zip(y_preds, y_trues.T, feature_processors)]
    loss = sum(losses)
    return loss


def train_model(autoencoder, optimizer, dataloader, criterion, epochs, device, feature_processors):
    logging.info(f'Start training')

    n_features = len(feature_processors)
    n_batches = len(dataloader)
    n_losses = 4

    train_losses = np.zeros((n_epochs, n_losses))

    for epoch in tqdm(range(epochs)):
        autoencoder.train()
        train_losses_per_epoch = np.zeros(n_losses)
        for images_batch, parameters_batch in dataloader:
            optimizer.zero_grad()
            reconstruction, mean, log_var, latent_sample, feature_discr = autoencoder(images_batch.to(device))
            loss = criterion(images_batch.to(device).float(), reconstruction,
                             mean, log_var,
                             feature_discr, parameters_batch[:, 1:].to(device), feature_processors)
            loss[0].backward()

            optimizer.step()

            for i in range(n_losses):
                train_losses_per_epoch[i] += loss[i].item() / n_batches

        print(train_losses_per_epoch)

        train_losses[epoch] = train_losses_per_epoch

        look_on_results(autoencoder, dataloader, device)
    return train_losses


def look_on_results(autoencoder, dataloader, device, n_to_show=6):
    autoencoder.eval()
    with torch.no_grad():
        for images_batch, parameters_batch in dataloader:
            reconstruction, mean, log_var, latent_sample, feature_discr = autoencoder(images_batch.to(device))
            result = reconstruction.cpu().detach().numpy()
            ground_truth = images_batch.numpy()
            break

    plt.figure(figsize=(8, 20))
    for i, (gt, res) in enumerate(zip(ground_truth[:n_to_show], result[:n_to_show])):
        plt.subplot(n_to_show, 2, 2 * i + 1)
        plt.imshow(gt.transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_to_show, 2, 2 * i + 2)
        plt.imshow(res.transpose(1, 2, 0), cmap='gray')

    plt.show()


def save_model(autoencoder: VAE, path: str = './model.pt') -> None:
    torch.save(autoencoder.state_dict(), path)


def load_model(path: str = './model.pt', latent_dim: int = 5, feature_processors=None) -> VAE:
    model: VAE = VAE(latent_dim=latent_dim, feature_processors=feature_processors)
    model.load_state_dict(torch.load(path))
    return model


def plt_images(*images):
    plt.figure(figsize=(20, 8))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()


def plt_latent_operations(model, x1, x2, n_features):
    plt.figure(figsize=(40, 40))

    sampled_x1 = model.sample_latent(x1)
    sampled_x2 = model.sample_latent(x2)

    decoded_x1 = model.decode_latent(sampled_x1)
    decoded_x2 = model.decode_latent(sampled_x2)

    plt_images(x1, decoded_x1, x2, decoded_x2)

    properties = choices(list(product([True, False], repeat = 6)), k=5)
    n_features = len(properties)

    for i in range(n_features):

        result_vector = model.latent_operations(x1, x2, properties[i])
        plt.subplot(n_features, 3, 3 * i + 1)
        plt.imshow(x1.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_features, 3, 3 * i + 2)
        plt.imshow(result_vector.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
        plt.subplot(n_features, 3, 3 * i + 3)
        plt.imshow(x2.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()



if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    RESUME_TRAINING = True

    latent_dim = 1024
    shape_processor = FeatureClassifier(latent_dim, 3)
    x_processor = FeatureRegressor(latent_dim)
    y_processor = FeatureRegressor(latent_dim)
    size_processor = FeatureRegressor(latent_dim)
    rotate_processor = FeatureRegressor(latent_dim)
    feature_processors: List[Union[FeatureClassifier, FeatureRegressor]] = [processor.to(device) for processor in
                                                                            [shape_processor, x_processor,
                                                                             y_processor,
                                                                             size_processor, rotate_processor]]
    lr = 0.001
    n_epochs = 15
    n_features = len(feature_processors)

    logging.info(f"Device: {device}")
    logging.info(f"Epochs: {n_epochs}")
    logging.info(f"Latent dim: {latent_dim}")
    logging.info(f'Creating model')

    # ------------------------------------------------------------
    # train
    # ------------------------------------------------------------
    autoencoder = load_model(latent_dim=latent_dim, feature_processors=feature_processors) if RESUME_TRAINING else VAE(
        latent_dim=latent_dim, feature_processors=feature_processors)
    model = autoencoder.to(device)


    criterion = vae_loss
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    logging.info(f'Setting up dataloader')
    dataloader = get_dataloader('dsprites')

    losses = train_model(autoencoder, optimizer, dataloader, criterion, n_epochs, device, feature_processors)
    save_model(autoencoder)

    # ------------------------------------------------------------
    # test
    # ------------------------------------------------------------

    dataloader = get_dataloader('dsprites', batch_size=2)

    image, values = next(iter(dataloader))
    x1 = image[:1]
    x2 = image[1:2]

    x1 = x1.to(device)
    x2 = x2.to(device)

    sampled_x1 = model.sample_latent(x1)
    sampled_x2 = model.sample_latent(x2)

    decoded = []
    for i in range(5):
        decoded_x = model.decode_latent(sampled_x1[:, i], reduce=False)
        decoded.append(decoded_x)
    for i in range(5):
        decoded_x = model.decode_latent(sampled_x2[:, i], reduce=False)
        decoded.append(decoded_x)
    plt_images(*decoded)


    plt_latent_operations(model, x1, x2, n_features)



    # look_on_results(model, dataloader, device)

    print("Done")
