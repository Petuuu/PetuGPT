import config as C
import tiktoken
import torch
import torch.nn.functional as F
import matplotlib as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from src.build.model import GPTModel
from src.utils.dataloaders import SpamDataset


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


def calc_loss_batch(input_batch, target_batch, model, device=C.DEVICE):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return F.cross_entropy(logits, target_batch)  # loss


def calc_loss_loader(data_loader, model, device=C.DEVICE, n_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif n_batches is None:
        n_batches = len(data_loader)
    else:
        n_batches = min(n_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < n_batches:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / n_batches


def evaluate_model(model, train_loader, val_loader, eval_iter, device=C.DEVICE):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, n_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, n_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    n_epochs,
    eval_freq,
    eval_iter,
    device=C.DEVICE,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(
                    f"{f'epoch {epoch + 1:03d} (step {global_step:05d})':<23} | "
                    f"{f'train_loss={train_loss:.3f}':<16} | "
                    f"val_loss={val_loss:.3f}"
                )

        train_acc = calc_accuracy_loader(train_loader, model, n_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, n_batches=eval_iter)
        print(f"train_acc={train_acc*100:.2f} | val_acc={val_acc*100:.2f}")
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot(epochs, train_losses, val_losses, train_accs, val_accs, examples_seen):
    if isinstance(epochs, int) or len(epochs) != len(val_losses):
        epochs = list(range(1, len(val_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.plot(epochs, train_accs, label="Training accuracy")
    ax1.plot(epochs, val_accs, linestyle="-.", label="Validation accuracy")

    ax1.set_xlabel("Checkpoints")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator())

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(
        [
            f"{int(examples_seen[min(i, len(examples_seen)-1)])}"
            for i in range(len(ax1.get_xticks()))
        ]
    )
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig("classification-plot.png")
    plt.show()


def classify(
    text, model, tokenizer, device=C.DEVICE, max_length=None, pad_token_id=50256
):
    model.eval()

    supported_contect_length = model.pos_emb.weight.shape[1]
    input_ids = tokenizer.encode(text)[: min(max_length, supported_contect_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.as_tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    label_pred = torch.argmax(logits, dim=-1).item()

    return "spam" if label_pred == 1 else "not spam"


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
    model = GPTModel().to(C.DEVICE)
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

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, n_batches=5)
        val_loss = calc_loss_loader(val_loader, model, n_batches=5)
        test_loss = calc_loss_loader(test_loader, model, n_batches=5)

    print("BEFORE FINE-TUNING\n")

    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    n_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model, train_loader, val_loader, optimizer, n_epochs, eval_freq=50, eval_iter=5
    )
    plot(n_epochs, train_losses, val_losses, train_accs, val_accs, examples_seen)

    train_acc = calc_accuracy_loader(train_loader, model, n_batches=10)
    val_acc = calc_accuracy_loader(val_loader, model, n_batches=10)
    test_acc = calc_accuracy_loader(test_loader, model, n_batches=10)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, n_batches=5)
        val_loss = calc_loss_loader(val_loader, model, n_batches=5)
        test_loss = calc_loss_loader(test_loader, model, n_batches=5)

    print("\nAFTER FINE-TUNING\n")

    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc*100:.2f}%")

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    print("\nTESTING\n")

    text_1 = (
        "Hey, just wanted to check if we're still on for dinner tonigh? Let me know!"
    )
    class_1 = classify(text_1, model, tokenizer, max_length=train_dataset.max_length)
    text_2 = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award"
    class_2 = classify(text_2, model, tokenizer, max_length=train_dataset.max_length)

    print(f"\nText: {text_1}\nClassification: {class_1}")
    print(f"\nText: {text_2}\nClassification: {class_2}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        C.CLASSIFICATION_MODEL_FILE,
    )
