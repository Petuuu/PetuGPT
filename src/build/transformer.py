import config as C
import torch
import torch.nn as nn
from src.build.attention import MultiHeadAttentionLayer
from src.build.MLP import LayerNorm, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNorm()
        self.att = MultiHeadAttentionLayer()
        self.ff = FeedForward()
        self.norm2 = LayerNorm()
        self.dropout = nn.Dropout(C.DROPOUT)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += shortcut
        return x


if __name__ == "__main__":
    x = torch.rand(2, 4, 768)
    block = TransformerBlock()
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print()
    print(sum(p.numel() for p in block.ff.parameters()))
