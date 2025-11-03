# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# %%
class NetGirl(nn.Module):
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


# def forward(inp, l1: nn.Linear, l2: nn.Linear):
#     u1 = l1.forward(inp)
#     s1 = F.tanh(u1)

#     u2 = l2.forward(s1)
#     s2 = F.tanh(u2)

    
# %%
model = NetGirl(3,2,1)

print(model)
gen_p = model.parameters()
print(list(gen_p))
# %%
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
# %%
model.train() # just good practice, not require for this model

# %%
x_train = torch.randn(8,3).float()
y_train = torch.where(torch.rand(8) < 0.5, -1, 1).float()
total = len(y_train)
y_train
# %%
from random import randint

for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k])
    loss = loss_func(y, y_train[k])

    # one step of SGD (stohastic gradient descinct)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# %%
model.eval()
# %%
for x, d in zip(x_train, y_train):
    with torch.no_grad():
        y = model(x)
        print(f"Выходные значения Нейронки: {y.data} => {d}")
# %%

