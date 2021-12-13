import torch

latent_dim = 3
n_features = 2
batch_size = 3

a = torch.Tensor(range(batch_size * latent_dim * n_features))
print(f'a')
print(a)

b = a.view(-1, n_features, latent_dim)

print('b')
print(b)

feat_1, feat_2 = b.unbind(1)
print(feat_1)
print(feat_2)

