import config as C
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        tokens = tokenizer.encode(txt)
        windows = tokens.unfold(0, max_length + 1, stride)

        self.input_ids = windows[:, :-1]
        self.target_ids = windows[:, 1:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(t) for t in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [t[: self.max_length] for t in self.encoded_texts]

        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t)) for t in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def _longest_encoded_length(self):
        max_length = 0
        for t in self.encoded_texts:
            t_length = len(t)
            if t_length > max_length:
                max_length = t_length
        return max_length


def create_dataloader(
    txt,
    tokenizer,
    batch_size=128,
    max_length=C.CONTEXT_LEN,
    stride=C.CONTEXT_LEN,
    shuffle=True,
    drop_last=True,
    n_workers=C.CORES,
    distributed=False,
):
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    sampler = None if not distributed else DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        drop_last=drop_last,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    from src.build.tokenizer import BPETokenizer, TOKENIZER_DATA

    tokenizeri = BPETokenizer()
    dataloader = create_dataloader(
        "<|endoftext|>".join(TOKENIZER_DATA[:5]),
        tokenizer=tokenizeri,
        batch_size=8,
        max_length=4,
        stride=4,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
