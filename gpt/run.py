import config as C
import sys, argparse
import torch
import tiktoken
from time import time
from src.build.tokenizer import BPETokenizer
from src.build.model import GPTModel
from src.pretrain_finetune.openai_weights import text_to_tokens, tokens_to_text
from src.utils.dataloaders import format_input


def generate_with_local_pretraining(text):
    checkpoint = torch.load(C.MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = BPETokenizer()
    tokens = model.generate(
        idx=text_to_tokens(text, tokenizer).to(C.DEVICE),
        max_tokens=25,
        top_k=20,
        temp=1.5,
    )
    print("Output text:\n", tokens_to_text(tokens[0], tokenizer))


def generate_with_openai_weights(text):
    checkpoint = torch.load(C.OPENAI_WEIGHTS_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = model.generate(
        idx=text_to_tokens(text, tokenizer).to(C.DEVICE),
        max_tokens=25,
        top_k=20,
        temp=1.5,
    )
    print("Output text:\n", tokens_to_text(tokens[0], tokenizer))


def classify(text, pad_token_id=50256):
    checkpoint = torch.load(C.CLASSIFICATION_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.out_head = torch.nn.Linear(in_features=C.EMB_DIM, out_features=2).to(C.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    supported_contect_length = model.pos_emb.weight.shape[1]
    input_ids = tokenizer.encode(text)[:supported_contect_length]
    input_ids += [pad_token_id] * (supported_contect_length - len(input_ids))
    input_tensor = torch.as_tensor(input_ids, device=C.DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    label_pred = torch.argmax(logits, dim=-1).item()
    ans = "spam" if label_pred == 1 else "not spam"

    print(f"Text is {ans}.")


def follow_instruction(text, input_):
    checkpoint = torch.load(C.ASSISTANT_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    form = format_input({"instruction": text, "input": input_})
    encoded = torch.as_tensor(tokenizer.encode(form), dtype=torch.long)
    encoded = encoded.unsqueeze(0).to(C.DEVICE)

    with torch.no_grad():
        token_ids = model.generate(encoded, 50, C.CONTEXT_LEN, temp=2, top_k=25)
    decoded = tokenizer.decode(token_ids[0].squeeze(0).tolist())
    print(decoded.replace("\n", " "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local-pretrain", action="store_true")
    parser.add_argument("-c", "--classification", action="store_true")
    parser.add_argument("-a", "--assistant", action="store_true")
    arguments = parser.parse_args(sys.argv[1:])
    text = input("Prompt: ")

    start = time()
    if arguments.local_pretrain:
        generate_with_local_pretraining(text)
    elif arguments.classification:
        classify(text)
    elif arguments.assistant:
        input_ = input("Input: ")
        follow_instruction(text, input_)
    else:
        generate_with_openai_weights(text)
    end = time()
    print(end - start)
