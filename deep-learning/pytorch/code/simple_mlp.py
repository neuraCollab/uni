# A minimal 2-layer MLP trained with a manual stochastic loop (one random
# sample per step, rather than DataLoader batching) - the smallest possible
# end-to-end PyTorch training example: model -> loss -> backward -> step.
#
# Ported from the archive's linModule.py, refactored:
# - class renamed from the placeholder `NetGirl` to `SimpleMLP`
# - removed a block of dead commented-out code (a manual forward() written
#   out by hand before the author switched to nn.Module/autograd)
# - fixed a stale comment: it said "one step of SGD (stochastic gradient
#   descent)" but the optimizer actually constructed and used is
#   `optim.RMSprop`, not SGD - comment now matches the code.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint


# %%
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x


# %%
model = SimpleMLP(3, 2, 1)
print(model)
print(list(model.parameters()))

# %%
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# %%
model.train()  # sets train-mode behavior for layers like Dropout/BatchNorm
               # (no effect here since this model has none, but good practice)

x_train = torch.randn(8, 3).float()
y_train = torch.where(torch.rand(8) < 0.5, -1, 1).float()
total = len(y_train)

# %% Manual stochastic training loop: one random sample per iteration
for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k])
    loss = loss_func(y, y_train[k])

    # one step of RMSprop (adaptive learning rate via a running average of
    # squared gradients - see fundamentals/optimization-sgd-adam.md)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% Evaluation
model.eval()
for x, target in zip(x_train, y_train):
    with torch.no_grad():
        y = model(x)
        print(f"Model output: {y.data} => target: {target}")
