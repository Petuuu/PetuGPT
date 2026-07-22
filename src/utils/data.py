import config as C


def load_tokenizer_data():
    with open(C.TOKENIZER_FILE, "r", encoding="utf-8") as f:
        raw = f.readlines()
    return [l.rstrip() for l in raw]


def load_sample_data():
    with open(C.SAMPLE_FILE, "r", encoding="utf-8") as f:
        print("Sample file read")
        return f.readlines()


def load_pretraining_data():
    with open(C.PRETRAINING_FILE, "r", encoding="utf-8") as f:
        raw = f.readlines()
    return [l.rstrip() for l in raw]
