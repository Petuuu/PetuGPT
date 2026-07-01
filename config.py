import os
import torch

torch.manual_seed(1009)

VOCAB_SIZE = 38338
CONTEXT_LEN = 4096
EMB_DIM = 1024
N_HEADS = 16
N_LAYERS = 24
DROPOUT = 0.1
QKV_BIAS = False

CORES = 10
SAMPLE_BYTES = 30 * 1024 * 1024
TOKENIZER_BYTES = 500 * 1024 * 1024
PRETRAINING_GB = 10 * 1024 * 1024 * 1024
SAMPLE_FILE = "data\\academic_sample.txt"
TOKENIZER_FILE = "data\\academic_tokenizing.txt"
TOKENIZER_CONFIG = "data\\tokenizer_config.txt"
PRETRAINING_FILE = "data\\academic_pretraining.txt"
DATA = [SAMPLE_FILE, SAMPLE_BYTES]
