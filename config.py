import os

VOCAB_SIZE = 5000
SAMPLE_BYTES = 30 * 1024 * 1024
TOKENIZER_BYTES = 500 * 1024 * 1024
PRETRAINING_GB = 10 * 1024 * 1024 * 1024
SAMPLE_FILE = "data\\academic_sample.txt"
TOKENIZER_FILE = "data\\academic_tokenizing.txt"
PRETRAINING_FILE = "data\\academic_pretraining.txt"
to_use = [SAMPLE_FILE, SAMPLE_BYTES]
