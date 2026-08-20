import config as C
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
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


def format_input(entry):
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
        f"\n\n### Input:\n{entry['input'] if entry['input'] else ''}"
    )


class InstructionDataset(Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        train_ratio=0.85,
        test_ratio=0.1,
    ):

        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.encoded_instructions = [
            tokenizer.encode(
                format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            )
            for entry in self.data
        ]

        train_portion = int(len(self.data) * train_ratio)
        test_portion = int(len(self.data) * test_ratio)

        self.train_data = self.encoded_instructions[:train_portion]
        self.test_data = self.encoded_instructions[
            train_portion : train_portion + test_portion
        ]
        self.val_data = self.encoded_instructions[train_portion + test_portion :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_instructions[idx]


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


def instruction_collate(
    batch, pad_token_id=50256, ignore_index=-100, max_length=None, device="cpu"
):

    batch_max_len = max(len(i) + 1 for i in batch)
    inputs_lst, targets_lst = [], []

    for i in batch:
        newi = i.copy()
        newi += [pad_token_id]
        padded = newi + [pad_token_id] * (batch_max_len - len(newi))

        inputs = torch.as_tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if max_length is not None:
            inputs = inputs[:max_length]
            targets = targets[:max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    worker_info = get_worker_info()
    if worker_info is not None and torch.device(device).type == "cuda":
        return inputs_tensor, targets_tensor

    inputs_tensor = inputs_tensor.to(device)
    targets_tensor = targets_tensor.to(device)
    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = InstructionDataset(C.INSTRUCTION_FILE, tokenizer)
    batch = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
    print(instruction_collate(batch))
