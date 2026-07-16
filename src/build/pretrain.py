import config as C
import torch
import torch.nn.functional as F
from src.build.tokenizer import BPETokenizer
from src.build.model import GPTModel
from src.utils.dataloader import create_dataloader
from src.utils.data import load_sample_data


def calc_loss_batch(input_batch, target_batch, model, device=C.DEVICE):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())  # loss


def calc_loss_loader(data_loader, model, device=C.DEVICE, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        print(i)
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    sample_data = load_sample_data()
    full = "<|endoftext|>".join(sample_data[:200])
    split_idx = int(C.TRAIN_RATIO * len(full))

    train_data, val_data = full[:split_idx], full[split_idx:]
    train_loader = create_dataloader(
        train_data, tokenizer=tokenizer, batch_size=4, num_workers=4
    )
    val_loader = create_dataloader(
        val_data, tokenizer=tokenizer, batch_size=4, num_workers=4
    )
    print(f"Train loader - {len(train_loader)} batches:")
    print(f"Validation loader - {len(val_loader)} batches")

    model = GPTModel()
    model.to(C.DEVICE)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model)
        val_loss = calc_loss_loader(val_loader, model)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


#### JUNK !!!

if __name__ == "_main__":
    tokenizer = BPETokenizer()
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a picos"
    batch = [t.unsqueeze(0) for t in tokenizer.encode(txt1, txt2)]
    batch = torch.stack(batch, dim=0)
    print(batch)
    print(batch.shape)
    print()

    torch.set_printoptions(sci_mode=False)
    model = GPTModel()
    model.eval()
    for out in model.generate(batch, 6):
        print("Output:", out)
        print("Decoded:", tokenizer.decode(out.squeeze(0)))
        print()

if __name__ == "_main__":
    tokenizer = BPETokenizer()
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a picos"
    batch = tokenizer.encode(txt1, txt2)
    batch = torch.stack(batch, dim=0)
    print(batch)
    print(batch.shape)
    print()

    model = GPTModel()
    torch.set_printoptions(sci_mode=False)
    with torch.no_grad():
        logits = model(batch)

    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    print(probas.shape)
    print(probas)
    print(idx_next)
