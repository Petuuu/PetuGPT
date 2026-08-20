import config as C
import tiktoken
import torch
from functools import partial
from torch.utils.data import DataLoader
from src.build.model import GPTModel
from src.utils.dataloaders import InstructionDataset, format_input, instruction_collate
from src.pretrain_finetune.pretrain import (
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    plot_losses,
)


def generate_and_print_sample(model, tokenizer, start_context, device, temp, top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = torch.tensor(tokenizer.encode(start_context), dtype=torch.long)
    encoded = encoded.unsqueeze(0).to(device)
    with torch.no_grad():
        token_ids = model.generate(encoded, 50, context_size, temp=temp, top_k=top_k)
    decoded = tokenizer.decode(token_ids[0].squeeze(0).tolist())
    print(decoded.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    n_epochs,
    eval_freq,
    eval_iter,
    start_context,
    device=C.DEVICE,
    temp=2,
    top_k=25,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(n_epochs):
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
                    f"{f'epoch {epoch + 1:03d} (step {global_step:05d})':<23} | "
                    f"{f'train_loss={train_loss:.3f}':<16} | "
                    f"val_loss={val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, start_context, device, temp, top_k)

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    collate = partial(instruction_collate, device=C.DEVICE, max_length=1024)

    instruction_dataset = InstructionDataset(
        json_file=C.INSTRUCTION_FILE, tokenizer=tokenizer
    )
    train_loader = DataLoader(
        dataset=instruction_dataset.train_data,
        batch_size=C.BATCH_SIZE,
        collate_fn=collate,
        shuffle=True,
        num_workers=C.CORES,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=instruction_dataset.val_data,
        batch_size=C.BATCH_SIZE,
        collate_fn=collate,
        num_workers=C.CORES,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=instruction_dataset.test_data,
        batch_size=C.BATCH_SIZE,
        collate_fn=collate,
        num_workers=C.CORES,
        drop_last=False,
    )

    checkpoint = torch.load(C.OPENAI_WEIGHTS_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel().to(C.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    n_epochs = 5

    example = {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
    }

    plot_losses(
        n_epochs,
        *train_model_simple(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            n_epochs=n_epochs,
            eval_freq=5,
            eval_iter=5,
            start_context=format_input(example),
        ),
    )

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, n_batches=5)
        val_loss = calc_loss_loader(val_loader, model, n_batches=5)
        test_loss = calc_loss_loader(test_loader, model, n_batches=5)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        C.INSTRUCTION_MODEL_FILE,
    )
