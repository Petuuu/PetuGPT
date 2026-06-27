from src.build.tokenizer import BPEtokenizer
from src.build.dataloader import create_dataloader
from src.build.embedding_layer import EmbeddingLayer
from src.build.attention import MultiHeadAttentionLayer
from src.utils.data import load_sample_data

if __name__ == "__main__":
    SAMPLE_DATA = load_sample_data()
    tokenizeri = BPEtokenizer()
    embedding = EmbeddingLayer()
    dataloader = create_dataloader(
        "<|endoftext|>".join(SAMPLE_DATA),
        tokenizer=tokenizeri,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, _ = next(data_iter)
    embeddings = embedding.compute_embeddings(inputs)
    print(inputs)
    print(embeddings)
    print(embeddings.shape)
