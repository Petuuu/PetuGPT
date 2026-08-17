import config as C
import numpy as np
import torch
import torch.nn as nn
import tiktoken
from src.utils.gpt_download import download_and_load_gpt2
from src.build.model import GPTModel


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.as_tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def tokens_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())


def assign(l, r):
    if l.shape != r.shape:
        raise ValueError(f"Shape mismatch. Left: {l.shape}, Right: {r.shape}")
    return nn.Parameter(torch.tensor(r))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# save model
if __name__ == "_main__":
    settings, params = download_and_load_gpt2(
        model_size=C.OPENAI_MODEL_SIZE, models_dir="gpt2"
    )
    gpt = GPTModel()
    load_weights_into_gpt(gpt, params)
    gpt.to(C.DEVICE)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = gpt.generate(
        idx=text_to_tokens("Every effort moves you", tokenizer).to(C.DEVICE),
        max_tokens=25,
        top_k=50,
        temp=1.5,
    )
    print("Output text:\n", tokens_to_text(tokens[0], tokenizer))

    torch.save(
        {
            "model_state_dict": gpt.state_dict(),
        },
        C.OPENAI_WEIGHTS_MODEL_FILE,
    )

# load and use model from file
if __name__ == "__main__":
    checkpoint = torch.load(C.OPENAI_WEIGHTS_MODEL_FILE, map_location=C.DEVICE)
    gpt = GPTModel()
    gpt.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = gpt.generate(
        idx=text_to_tokens("The capital of france is", tokenizer).to(C.DEVICE),
        max_tokens=25,
        top_k=50,
        temp=1.5,
    )
    print("Output text:\n", tokens_to_text(tokens[0], tokenizer))
