# %% 
import torch
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def act(x):
    return 0 if x < 0.5 else 1
# %%
def go(house, rock, attr):
    X = torch.tensor([house, rock, attr], dtype=torch.float32)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])
    Wout = torch.tensor([-1.0, 1.0])

    Zh = torch.mv(Wh, X)

    Uh = torch.tensor([act(x) for x in Zh], dtype=torch.float32)

    Zout = torch.dot(Wout, Uh)

    Y = act(Zout)
    return (Y)

# %%
house = 1
rock = 0
attr = 1
# %%
out = go(house=house, rock=rock, attr=attr)
# %%
print("yes") if out == 1 else print("no")
# %%
