import torch

# f = w * x 

# f = 2 * x
X = torch.tensor([1, 2, 3], dtype=torch.float32)
Y = torch.tensor([2, 4, 6], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
  return w * x

# calculate loss. loss = Mean Squared Error
def loss(y, y_predicted):
  return ((y_predicted - y) ** 2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 35

for epoch in range(n_iters):
  # prediction
  y_pred = forward(X)

  # loss
  l = loss(Y, y_pred)

  # gradients
  l.backward() #dl/dw - collects in w.grad

  # update weights (update formula)
  # should not make this part of our computational graph
  with torch.no_grad():
    w -= learning_rate * w.grad

  # zero out gradients because they accumulate
  w.grad.zero_()

  if epoch % 1 == 0:
    print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')