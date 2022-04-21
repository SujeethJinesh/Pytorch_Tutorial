from tkinter import Y
import torch
import numpy as np

x = torch.tensor(1, dtype=torch.float32)
y = torch.tensor(2, dtype=torch.float32)
w = torch.tensor(1, requires_grad=True, dtype=torch.float32)

# forward pass
y_hat = x * w
s = y_hat - y
loss = s*s

print(loss)

# backwards pass
loss.backward()
print(w.grad)

#update weights
# next forward and backward pass


