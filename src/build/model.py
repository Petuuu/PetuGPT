import config as C
import torch
import torch.nn as nn
from src.build.tokenizer import BPEtokenizer
from src.utils.dataloader import create_dataloader
from src.build.attention import MultiHeadAttentionLayer
from src.build.transformer import TransformerBlock
from src.build.MLP import LayerNorm


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(C.VOCAB_SIZE, C.EMB_DIM)
        self.pos_emb = nn.Embedding(C.CONTEXT_LEN, C.EMB_DIM)
        self.drop_emb = nn.Dropout(C.DROPOUT)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(C.N_LAYERS)]
        )
        self.final_norm = LayerNorm()
        self.out_head = nn.Linear(C.EMB_DIM, C.VOCAB_SIZE, bias=False)
        print("Model initialized")

    def forward(self, idx):
        _, seq_len = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=idx.device))

        x = self.drop_emb(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)  # logits

    def generate(self, idx, max_tokens, context_len=C.CONTEXT_LEN):
        for _ in range(max_tokens):
            idx_cond = idx[:, -context_len:]
            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch = tokenizer.encode(txt1, txt2)
    batch = tokenizer.pad1d(batch)
    batch = torch.stack(batch, dim=0)

    start = "Hello, I am"
    encoded = tokenizer.encode(start).unsqueeze(0)
    print("Encoded:", encoded)

    model = GPTModel()
    model.eval()
    out = model.generate(encoded, 6)

    torch.set_printoptions(sci_mode=False)
    print("Output:", out)
    print("Output length:", len(out[0]))
    print("Decoded:", tokenizer.decode(out.squeeze(0)))

    # tok_emb = sum(p.numel() for p in model.tok_emb.parameters())
    # pos_emb = sum(p.numel() for p in model.pos_emb.parameters())
    # trf_blocks = sum(p.numel() for p in model.trf_blocks.parameters())
    # final_norm = sum(p.numel() for p in model.final_norm.parameters())
    # params = sum(p.numel() for p in model.parameters())
    # print(f"Token embedding & output layer parameters: {tok_emb:,}")
    # print(f"Position embedding parameters: {pos_emb:,}")
    # print(f"Transformer parameters: {trf_blocks:,}")
    # print(f"Final normalization parameters: {final_norm:,}")
    # print(f"\nTotal number of parameters: {params:,}")
