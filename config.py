import os
import torch

torch.manual_seed(1009)

CORES = 10
VOCAB_SIZE = 5000
SAMPLE_BYTES = 30 * 1024 * 1024
TOKENIZER_BYTES = 500 * 1024 * 1024
PRETRAINING_GB = 10 * 1024 * 1024 * 1024
SAMPLE_FILE = "data\\academic_sample.txt"
TOKENIZER_FILE = "data\\academic_tokenizing.txt"
TOKENIZER_CONFIG = "data\\tokenizer_settings.txt"
PRETRAINING_FILE = "data\\academic_pretraining.txt"
DATA = [SAMPLE_FILE, SAMPLE_BYTES]
