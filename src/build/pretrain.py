import config as C
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


def evaluate_model(model, train_loader, val_loader, eval_iter, device=C.DEVICE):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context, device):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context).to(device)
    with torch.no_grad():
        token_ids = model.generate(encoded, 50, context_size)
    decoded = tokenizer.decode(token_ids)
    print(decoded.replace("\n", " "))
    model.train()


def train_model(
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    device=C.DEVICE,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"{f'epoch {epoch + 1:03d} (step {global_step:06d})':<28} | "
                    f"{f'train_loss={train_loss:.3f}':<18} | "
                    f"{f'val_loss={val_loss:.3f}':<18}"
                )

        generate_and_print_sample(model, tokenizer, start_context, device)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs, train_losses, val_losses, tokens_seen):
    if isinstance(epochs, int) or len(epochs) != len(val_losses):
        epochs = list(range(1, len(val_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator())
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    plot_losses(
        num_epochs,
        *train_model(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            eval_freq=5,
            eval_iter=5,
            start_context="Every effort moves you",
        ),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        C.MODEL_FILE,
    )
