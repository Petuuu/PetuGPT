import os
import torch

torch.manual_seed(1009)

VOCAB_SIZE = 45098
CONTEXT_LEN = 2048
EMB_DIM = 1024
N_HEADS = 16
N_LAYERS = 24
DROPOUT = 0.1
QKV_BIAS = False
TRAIN_RATIO = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORES = 10
SAMPLE_BYTES = 10 * 1024 * 1024
TOKENIZER_BYTES = 30 * 1024 * 1024
PRETRAINING_GB = 1024 * 1024 * 1024
SAMPLE_FILE = "data\\academic_sample.txt"
TOKENIZER_FILE = "data\\academic_tokenizing.txt"
TOKENIZER_CONFIG = "data\\tokenizer_config.txt"
PRETRAINING_FILE = "data\\academic_pretraining.txt"
MODEL_FILE = "data\\model.pth"
DATA = [TOKENIZER_FILE, TOKENIZER_BYTES]
