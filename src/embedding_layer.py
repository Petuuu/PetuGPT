import config as c
from src.dataloader import create_dataloader
from src.tokenizer import BPEtokenizer
from src.data import load_sample_data
import torch

SAMPLE_DATA = load_sample_data()
max_length = context_length = 4

token_embedding_layer = torch.nn.Embedding(5000, 768)
pos_embedding_layer = torch.nn.Embedding(5000, 768)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

if __name__ == "__main__":
    dataloader = create_dataloader(
        "<|endoftext|>".join(SAMPLE_DATA),
        tokenizer=BPEtokenizer(),
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Tokens:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print("Token embeddings shape:\n", token_embeddings.shape)
    print("Positional embeddings shape:\n", pos_embeddings.shape)
