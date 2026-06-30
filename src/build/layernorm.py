import config as C
import torch
import torch.nn as nn


class DummyLayerNorm(nn.Module):
    def __init__(self, dim=C.EMB_DIM):
        super().__init__()

    def forward(self, x):
        return x
