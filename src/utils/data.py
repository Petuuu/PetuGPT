import config as c


def load_tokenizer_data():
    with open(c.TOKENIZER_FILE, "r", encoding="utf-8") as f:
        raw = f.readlines()
    return [l.rstrip() for l in raw]


def load_sample_data():
    with open(c.SAMPLE_FILE, "r", encoding="utf-8") as f:
        print("Sample file read")
        return f.readlines()
