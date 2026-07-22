import config as C
import torch
import torch.nn as nn
from src.build.tokenizer import BPETokenizer
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

    def generate(
        self,
        idx,
        max_tokens,
        context_len=C.CONTEXT_LEN,
        temp=0.0,
        top_k=None,
        eos_id=0,
    ):
        while idx.dim() > 2 and idx.shape[0] == 1:
            idx = idx.squeeze(0)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)

        res = []
        for t in idx:
            for i in range(t.shape[0], 0, -1):
                if t[i - 1] != 0:
                    res.append(t[:i].unsqueeze(0))
                    break

        for b in range(len(res)):
            for _ in range(max_tokens):
                idx_cond = res[b][:, -context_len:]
                with torch.no_grad():
                    logits = self(idx_cond)
                logits = logits[:, -1, :]

                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    logits = torch.where(
                        logits < top_logits[:, -1],
                        torch.tensor(float("-inf")).to(logits.device),
                        logits,
                    )

                if temp > 0.0:
                    probas = torch.softmax(logits / temp, dim=-1)
                    idx_next = torch.multinomial(probas, num_samples=1)
                else:
                    probas = torch.softmax(logits, dim=-1)
                    idx_next = torch.argmax(probas, dim=-1, keepdim=True)

                if idx_next == eos_id:
                    break
                res[b] = torch.cat((res[b], idx_next), dim=1)

        return res


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    encoded = tokenizer.encode("Hello, I am").unsqueeze(0).unsqueeze(0)
    print("Encoded:", encoded)

    model = GPTModel()
    model.eval()
    torch.set_printoptions(sci_mode=False)
    for out in model.generate(encoded, 6):
        print("Output:", out)
        print("Output length:", len(out[0]))
        print("Decoded:", tokenizer.decode(out.squeeze(0)))
