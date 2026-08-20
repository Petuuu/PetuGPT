import config as C
import argparse, sys
import torch
import tiktoken
from time import time
from src.build.tokenizer import BPETokenizer
from src.build.model import GPTModel
from src.pretrain_finetune.openai_weights import text_to_tokens, tokens_to_text


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


def classify(text, max_length=None, pad_token_id=50256):
    checkpoint = torch.load(C.CLASSIFICATION_MODEL_FILE, map_location=C.DEVICE)
    model = GPTModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    supported_contect_length = model.pos_emb.weight.shape[1]
    input_ids = tokenizer.encode(text)[: min(max_length, supported_contect_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.as_tensor(input_ids, device=C.DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    label_pred = torch.argmax(logits, dim=-1).item()
    ans = "spam" if label_pred == 1 else "not spam"

    print(f"Text is {ans}.")


def follow_instruction(text):
    print("WIP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local-pretrain", action="store_true")
    parser.add_argument("-c", "--classification", action="store_true")
    parser.add_argument("-i", "--instruction", action="store_true")
    arguments = parser.parse_args(sys.argv[1:])
    text = input("Prompt: ")

    start = time()
    if arguments.local_pretrain:
        generate_with_local_pretraining(text)
    elif arguments.classification:
        classify(text)
    elif arguments.instruction:
        follow_instruction(text)
    else:
        generate_with_openai_weights(text)
    end = time()
    print(end - start)
