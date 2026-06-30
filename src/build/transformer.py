import config as C
import torch
import torch.nn as nn


class DummyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
