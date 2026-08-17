import config as C
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim=C.EMB_DIM):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, dim=C.EMB_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.layers(x)
