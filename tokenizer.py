from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import heapq
import ast
import re
from config import TOKENIZER_FILE, VOCAB_SIZE, CORES

with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
    raw = f.readlines()
TOKENIZER_DATA = [l.rstrip() for l in raw]

pattern = re.compile(
    r'<\|endoftext\|>|<\|unk\|>|---|--|...|..|\s*[A-Za-z0-9]+|[()[],._?!"\'-]'
)


def pre_tokenize(line):
    words = pattern.findall(line)
    return [w.replace(" ", "G̃") for w in words]


def _worker(lines):
    local = defaultdict(int)
    for line in lines:
        for w in pre_tokenize(line):
            local[w] += 1
    return local


def compute_word_freqs(data):
    print("Starting pre-tokenization...")

    chunk_size = (len(data) + CORES - 1) // CORES
    chunks_data = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    with Pool(cpu_count()) as pool:
        results = pool.map(_worker, chunks_data)

    print("Chunks done")

    word_freqs = defaultdict(int)
    for r in results:
        for k, v in r.items():
            word_freqs[k] += v

    print("Pre-tokenization done")

    return word_freqs


def base_vocab(word_freqs):
    alphabet = []
    for w in word_freqs.keys():
        for l in w:
            if l not in alphabet:
                alphabet.append(l)
    alphabet.sort()
    print("Base vocabulary initialized")
    return ["<|endoftext|>", "<|unk|>"] + alphabet.copy()


def init_structures(word_freqs):
    splits = {w: list(w) for w in word_freqs}

    pair_freqs = defaultdict(int)
    pair_positions = defaultdict(set)

    for w, f in word_freqs.items():
        tokens = splits[w]

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freqs[pair] += f
            pair_positions[pair].add((w, i))

    heap = [(-f, pair) for pair, f in pair_freqs.items()]
    heapq.heapify(heap)

    return splits, pair_freqs, pair_positions, heap


def merge(pair, splits, pair_freqs, pair_positions, heap, word_freqs):
    newi = "".join([*pair])
    affected = list(pair_positions[pair])
    pair_positions.pop(pair, None)

    for w, i in affected:
        tokens = splits[w]
        if i >= len(tokens) - 1 or tokens[i] != pair[0] or tokens[i + 1] != pair[1]:
            continue

        freq = word_freqs[w]

        if i > 0:
            l = (tokens[i - 1], tokens[i])
            pair_freqs[l] -= freq
            pair_positions[l].discard((w, i - 1))

        if i + 2 < len(tokens):
            r = (tokens[i + 1], tokens[i + 2])
            pair_freqs[r] -= freq
            pair_positions[r].discard((w, i + 1))

        tokens[i : i + 2] = [newi]

        if i > 0:
            lnew = (tokens[i - 1], newi)
            pair_freqs[lnew] += freq
            pair_positions[lnew].add((w, i - 1))
            heapq.heappush(heap, (-pair_freqs[lnew], lnew))

        if i < len(tokens) - 1:
            rnew = (newi, tokens[i + 1])
            pair_freqs[rnew] += freq
            pair_positions[rnew].add((w, i))
            heapq.heappush(heap, (-pair_freqs[rnew], rnew))


def create_vocab(data, vocab_size):
    word_freqs = compute_word_freqs(data)
    vocab = base_vocab(word_freqs)

    splits, pair_freqs, pair_positions, heap = init_structures(word_freqs)
    merges = {}

    while len(vocab) < vocab_size and heap:
        freq, pair = heapq.heappop(heap)
        freq = -freq

        if pair_freqs[pair] != freq or freq == 0:
            continue

        merge(pair, splits, pair_freqs, pair_positions, heap, word_freqs)

        newi = "".join([*pair])
        merges[pair] = newi
        vocab.append(newi)

        if len(vocab) % 2000 == 0:
            print("Vocab size:", len(vocab))

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    return token_to_id, id_to_token, vocab, merges


class BPEtokenizer:
    def __init__(self, data=TOKENIZER_DATA, vocab_size=VOCAB_SIZE):
        self.run = int(input("New (1) or backed up vocabulary (2)? "))
        if self.run == 1:
            config = create_vocab(data, vocab_size)
        elif self.run == 2:
            with open("data\\tokenizer_settings.txt", "r", encoding="utf-8") as f:
                config = ast.literal_eval(f.read())
        else:
            print("INVALID INPUT")
            exit()

        self.token_to_id, self.id_to_token, self.vocab, self.merges = config
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def encode_word(self, word):
        if word in {"<|endoftext|>", "<|unk|>"}:
            return [self.token_to_id[word]]

        tokens = list(word)
        if len(tokens) <= 1:
            return [self.token_to_id.get(word, 1)]

        while True:
            best_rank = None
            best_idx = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i

            if best_rank is None:
                break
            tokens[best_idx : best_idx + 2] = [tokens[best_idx] + tokens[best_idx + 1]]

        return tokens

    @lru_cache(maxsize=200_000)
    def encode_word_cached(self, word):
        return tuple(self.encode_word(word))

    def tokenize(self, text):
        words = pre_tokenize(text)
        tokens = []
        for w in words:
            tokens.extend(self.encode_word_cached(w))

        return list(tokens)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id.get(t, 1) for t in tokens]

    def decode(self, ids):
        text = "".join([self.id_to_token[x] for x in ids])
        return text.replace("G̃", " ")


if __name__ == "__main__":
    print("Cores available:", cpu_count())
    tokenizeri = BPEtokenizer()
    print(
        tokenizeri.tokenize(
            "The ([dominant]) sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
        )
    )
    ids = tokenizeri.encode(
        "The ([dominant]) sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
    )
    print(ids)
    print(len(ids))
    print(tokenizeri.decode(ids))
    print(len(tokenizeri.vocab))

    if tokenizeri.run == 1 and input("Save tokenizer configuration? [y/N]: ") in {
        "y",
        "Y",
    }:
        settings = [
            tokenizeri.token_to_id,
            tokenizeri.id_to_token,
            tokenizeri.vocab,
            tokenizeri.merges,
        ]
        with open("data\tokenizer_configuration.txt", "w", encoding="utf-8") as f:
            f.write(f"{settings}")
