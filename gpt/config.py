import os, sys
import json, argparse
import torch

settings = {
    "n_vocab": 45098,
    "n_ctx": 256,
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 24,
}
openai_dir = "gpt2/355M/"

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local-pretrain", action="store_true")
local = parser.parse_args(sys.argv[1:]).local_pretrain

if os.path.isdir(openai_dir) and not local:
    with open(openai_dir + "hparams.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

VOCAB_SIZE = settings["n_vocab"]
CONTEXT_LEN = settings["n_ctx"]
EMB_DIM = settings["n_embd"]
N_HEADS = settings["n_head"]
N_LAYERS = settings["n_layer"]
DROPOUT = 0.1
QKV_BIAS = False if VOCAB_SIZE == 45098 else True
TRAIN_RATIO = 0.9
BATCH_SIZE = 8
OPENAI_MODEL_SIZE = "355M"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORES = 10
SAMPLE_BYTES = 10 * 1024 * 1024
TOKENIZER_BYTES = 30 * 1024 * 1024
PRETRAINING_MB = 512 * 1024 * 1024

SAMPLE_FILE = "data/academic_sample.txt"
TOKENIZER_FILE = "data/academic_tokenizing.txt"
PRETRAINING_FILE = "data/academic_pretraining.txt"
TOKENIZER_CONFIG = "public/tokenizer_config.txt"
MODEL_FILE = "data/model.pth"
OPENAI_WEIGHTS_MODEL_FILE = "data/openai_weights_model.pth"
SPAM_FILE = "data/spam-data.txt"
SPAM_TRAIN_CSV = "public/spam-train.csv"
SPAM_VAL_CSV = "public/spam-val.csv"
SPAM_TEST_CSV = "public/spam-test.csv"
INSTRUCTION_FILE = "public/instruction-data.json"

DATA = [PRETRAINING_FILE, PRETRAINING_MB]
