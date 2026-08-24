import sys
import argparse
import torch

# parser = argparse.ArgumentParser()
# parser.add_argument("-l", "--local-pretrain", action="store_true")
# local = parser.parse_args(sys.argv[1:]).local_pretrain

if False:  # locally pretrained weights
    settings = {
        "n_vocab": 45098,
        "n_ctx": 256,
        "n_embd": 1024,
        "n_head": 16,
        "n_layer": 24,
    }
else:
    settings = {
        "n_vocab": 50257,
        "n_ctx": 1024,
        "n_embd": 1024,
        "n_head": 16,
        "n_layer": 24,
    }

# MODEL CONFIG

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

# DEVICE AND DOWNLOAD CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORES = 10
SAMPLE_BYTES = 10 * 1024 * 1024
TOKENIZER_BYTES = 30 * 1024 * 1024
PRETRAINING_MB = 512 * 1024 * 1024

# DATA FILES

SAMPLE_FILE = "data/academic_sample.txt"
PRETRAINING_FILE = "data/academic_pretraining.txt"
SPAM_FILE = "data/spam-data.txt"

# PUBLIC FILES

TOKENIZER_FILE = "public/academic_tokenizing.txt"
TOKENIZER_CONFIG = "public/tokenizer_config.txt"
SPAM_TRAIN_CSV = "public/spam-train.csv"
SPAM_VAL_CSV = "public/spam-val.csv"
SPAM_TEST_CSV = "public/spam-test.csv"
INSTRUCTION_FILE = "public/instruction-data.json"

# MODEL FILES

MODEL_FILE = "models/model.pth"
OPENAI_WEIGHTS_MODEL_FILE = "models/openai_weights_model.pth"
CLASSIFICATION_MODEL_FILE = "models/classifier_model.pth"
ASSISTANT_MODEL_FILE = "models/assistant_model.pth"

# OTHER

DATA = [PRETRAINING_FILE, PRETRAINING_MB]
