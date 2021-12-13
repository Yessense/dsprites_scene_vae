import torch


batch_size = 5
n_classes = 3
a = torch.randn((batch_size, n_classes ))
print(a)


b = torch.randint(4, (batch_size,))
c = torch.nn.functional.one_hot(b, n_classes)
print(b)
print(c)