import torch

a = torch.rand([4, 3, 3, 3])
print(a.shape)
b = a.index_select(0, torch.arange(0,3,2))
print(b.shape)
