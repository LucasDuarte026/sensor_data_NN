import numpy as np
import torch

data = torch.tensor([1,2,3,4,5,6,7,8,9,10])
data = torch.randn(1,2,3,4)
print(data.view(1,2,3,4))
