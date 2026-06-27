from tokenizer import BPEtokenizer, TOKENIZER_DATA
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        tokens = torch.tensor(tokenizer.encode(txt), dtype=torch.long)
        windows = tokens.unfold(0, max_length + 1, stride)
        print("Data encoded")

        self.input_ids = windows[:, :-1]
        self.target_ids = windows[:, 1:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    print("Dataset created")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    print("Dataloader created")
    return dataloader


if __name__ == "__main__":
    tokenizer = BPEtokenizer()
    print("Tokenizer initialized")
    dataloader = create_dataloader(
        "<|endoftext|>".join(TOKENIZER_DATA[:5]),
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False,
        num_workers=12,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
