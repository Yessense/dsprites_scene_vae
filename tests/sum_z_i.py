import torch

batch_size = 5
feature_count = 3
vector_size = 2

a = torch.tensor(range(batch_size * feature_count * vector_size))
print(a)

b = a.view(-1, feature_count, vector_size)
print(b)

c = torch.sum(b, 1)
print(c)