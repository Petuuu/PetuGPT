import config as C
import argparse, sys
import torch
import tiktoken
from src.build.tokenizer import BPETokenizer
from src.build.model import GPTModel
from time import time


def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.as_tensor(encoded, dtype=torch.long).unsqueeze(0)
    return encoded_tensor


def tokens_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    return tokenizer.decode(flat.tolist())


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
        pass
    elif arguments.instruction:
        pass
    else:
        generate_with_openai_weights(text)
    end = time()
    print(end - start)
