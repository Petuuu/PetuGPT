import config as C
import tiktoken
import torch
from torch.utils.data import DataLoader
from src.build.model import GPTModel
from src.utils.dataloaders import SpamDataset


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.as_tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def tokens_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_accuracy_loader(data_loader, model, device=C.DEVICE, n_batches=None):
    model.eval()
    correct_preds, n_examples = 0, 0
    n_batches = (
        len(data_loader) if n_batches is None else min(n_batches, len(data_loader))
    )

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < n_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            label_preds = torch.argmax(logits, dim=-1)

            n_examples += label_preds.shape[0]
            correct_preds += (label_preds == target_batch).sum().item()
        else:
            break

    return correct_preds / n_examples


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(csv_file=C.SPAM_TRAIN_CSV, tokenizer=tokenizer)
    val_dataset = SpamDataset(
        csv_file=C.SPAM_VAL_CSV,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    test_dataset = SpamDataset(
        csv_file=C.SPAM_TEST_CSV,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=True,
        num_workers=C.CORES,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=C.BATCH_SIZE,
        num_workers=C.CORES,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=C.BATCH_SIZE,
        num_workers=C.CORES,
        drop_last=False,
    )

    checkpoint = torch.load(C.OPENAI_WEIGHTS_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    n_class = 2
    model.out_head = torch.nn.Linear(in_features=C.EMB_DIM, out_features=n_class)

    train_acc = calc_accuracy_loader(train_loader, model, n_batches=10)
    val_acc = calc_accuracy_loader(val_loader, model, n_batches=10)
    test_acc = calc_accuracy_loader(test_loader, model, n_batches=10)

    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")
