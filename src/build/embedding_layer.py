import config as c
import torch


class EmbeddingLayer:
    def __init__(self, vocab_size=c.VOCAB_SIZE, dim=768):
        self.token_layer = torch.nn.Embedding(vocab_size, dim)
        self.pos_layer = torch.nn.Embedding(vocab_size, dim)
        print("Embedding layers initialized")

    def compute_embeddings(self, ids):
        return self.token_layer(ids) + self.pos_layer(ids)


if __name__ == "__main__":
    from src.build.dataloader import create_dataloader
    from src.build.tokenizer import BPEtokenizer
    from src.utils.data import load_sample_data

    SAMPLE_DATA = load_sample_data()
    max_length = context_length = 4

    embedding = EmbeddingLayer()
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

    embeddings = embedding.compute_embeddings
    print("Embeddings shape:\n", embeddings.shape)
