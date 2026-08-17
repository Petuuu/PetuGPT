import config as C
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.build.tokenizer import BPETokenizer
from src.build.model import GPTModel
from src.utils.dataloader import create_dataloader
from src.utils.data import load_pretraining_data


def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(C.os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


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
        if i < num_batches:
            with torch.amp.autocast("cuda", dtype=torch.float16):
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


def generate_and_print_sample(model, tokenizer, start_context, device, temp, top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context).to(device)
    with torch.no_grad():
        token_ids = model.generate(encoded, 50, context_size, temp=temp, top_k=top_k)
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
    scaler,
    device=C.DEVICE,
    temp=2,
    top_k=25,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    rank = dist.get_rank()

    for epoch in range(num_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tokens_seen += input_batch.numel() * dist.get_world_size()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                if rank == 0:
                    print(
                        f"{f'epoch {epoch + 1:03d} (step {global_step:05d})':<23} | "
                        f"{f'train_loss={train_loss:.3f}':<16} | "
                        f"val_loss={val_loss:.3f}"
                    )

            if global_step % 50 == 0 and global_step % eval_freq != 0 and rank:
                print(f"epoch {epoch + 1:03d} (step {global_step:05d})")

        if rank == 0:
            generate_and_print_sample(
                model.module, tokenizer, start_context, device, temp, top_k
            )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs, train_losses, val_losses, tokens_seen):
    if isinstance(epochs, int) or len(epochs) != len(val_losses):
        epochs = list(range(1, len(val_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")

    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator())

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(
        [
            f"{int(tokens_seen[min(i, len(tokens_seen)-1)])}"
            for i in range(len(ax1.get_xticks()))
        ]
    )
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()

    if rank == 0:
        print("CUDA AVAILABLE:", torch.cuda.is_available())
    tokenizer = BPETokenizer()
    pretraining_data = load_pretraining_data()
    full = "<|endoftext|>".join(pretraining_data)
    split_idx = int(C.TRAIN_RATIO * len(full))

    train_data, val_data = full[:split_idx], full[split_idx:]
    train_loader = create_dataloader(
        train_data,
        tokenizer=tokenizer,
        batch_size=C.BATCH_SIZE,
        num_workers=C.CORES,
        distributed=True,
    )
    val_loader = create_dataloader(
        val_data,
        tokenizer=tokenizer,
        batch_size=C.BATCH_SIZE,
        num_workers=C.CORES,
        distributed=False,
    )
    if rank == 0:
        print(f"Train loader - {len(train_loader)} batches:")
        print(f"Validation loader - {len(val_loader)} batches")

    model = GPTModel().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0002, betas=(0.9, 0.95), weight_decay=0.1
    )
    num_epochs = 10
    scaler = torch.amp.GradScaler("cuda")
    train_losses, val_losses, track_tokens_seen = train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_freq=150,
        eval_iter=10,
        start_context="Every effort moves you",
        scaler=scaler,
    )

    if rank == 0:
        plot_losses(num_epochs, train_losses, val_losses, track_tokens_seen)
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            C.MODEL_FILE,
        )

    dist.barrier()
    cleanup()
