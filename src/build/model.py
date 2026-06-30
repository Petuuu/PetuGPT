import config as C
import torch
import torch.nn as nn
from src.build.tokenizer import BPEtokenizer
from src.build.dataloader import create_dataloader
from src.build.attention import MultiHeadAttentionLayer
from src.build.transformer import DummyTransformerBlock
from src.build.layernorm import DummyLayerNorm


class DummyGPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(C.VOCAB_SIZE, C.EMB_DIM)
        self.pos_emb = nn.Embedding(C.CONTEXT_LEN, C.EMB_DIM)
        self.drop_emb = nn.Dropout(C.DROPOUT)
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock() for _ in range(C.N_LAYERS)]
        )
        self.final_norm = DummyLayerNorm(C.EMB_DIM)
        self.out_head = nn.Linear(C.EMB_DIM, C.VOCAB_SIZE, bias=False)

    def forward(self, idx):
        _, seq_len = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=idx.device))

        x = self.drop_emb(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)  # logits


if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = tokenizer.encode(txt1, txt2)
    batch = tokenizer.pad1d(batch)
    batch = torch.stack(batch, dim=0)

    model = DummyGPTModel()
    logits = model(batch)
    print("Shape:", logits.shape)
    print(logits)
