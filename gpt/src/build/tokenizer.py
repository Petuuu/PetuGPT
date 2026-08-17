import config as C
from src.utils.data import load_tokenizer_data
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import torch
import torch.nn.functional as F
import argparse
import heapq
import sys, ast, re

TOKENIZER_DATA = load_tokenizer_data()

pattern = re.compile(
    r'<\|endoftext\|>|<\|unk\|>|--|\s*[A-Za-z0-9]+|[\(\)\[\],._?!"\'—%&$€£-]'
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

    chunk_size = (len(data) + C.CORES - 1) // C.CORES
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
    print("Base vocabulary initialized, length:", len(alphabet) + 2)
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

        if len(vocab) % 1000 == 0:
            print("Vocab size:", len(vocab))

    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = {i: t for t, i in token_to_id.items()}

    return token_to_id, id_to_token, vocab, merges


class BPETokenizer:
    def __init__(self, data=TOKENIZER_DATA, vocab_size=C.VOCAB_SIZE):
        parser = argparse.ArgumentParser()
        parser.add_argument("-nv", "--new-vocab", action="store_true")
        new_vocab = parser.parse_args(sys.argv[1:]).new_vocab

        if new_vocab:
            config = create_vocab(data, vocab_size)
        else:
            with open("data/tokenizer_config.txt", "r", encoding="utf-8") as f:
                config = ast.literal_eval(f.read())

        self.token_to_id, self.id_to_token, self.vocab, self.merges = config
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if new_vocab and input("Save tokenizer configuration? [y/N]: ") == "y":
            settings = [self.token_to_id, self.id_to_token, self.vocab, self.merges]
            with open(C.TOKENIZER_CONFIG, "w", encoding="utf-8") as f:
                f.write(f"{settings}")

    def tokenize_word(self, word):
        tokens = list(word)
        if word in {"<|endoftext|>", "<|unk|>"} or len(tokens) <= 1:
            return [word]

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
    def tokenize_word_cached(self, word):
        return tuple(self.tokenize_word(word))

    def tokenize(self, text):
        words = pre_tokenize(text)
        tokens = []
        for w in words:
            tokens.extend(self.tokenize_word_cached(w))

        return list(tokens)

    def pad1d(self, tensors):
        maxi = max(t.shape[0] for t in tensors)

        if all(t.shape[0] == maxi for t in tensors):
            return tensors

        return [F.pad(t, (0, maxi - t.shape[0])) for t in tensors]

    def encode(self, *texts):
        if len(texts) == 0:
            raise TypeError(
                "At least 2 argument must be given to BPETokenizer.encode()"
            )

        tensors = [
            torch.tensor([self.token_to_id.get(t, 1) for t in self.tokenize(text)])
            for text in texts
        ]

        return tensors[0] if len(tensors) == 1 else self.pad1d(tensors)

    def decode(self, *tensors):
        if len(tensors) == 0:
            raise TypeError(
                "At least 2 argument must be given to BPETokenizer.decode()"
            )

        def to_ids(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().reshape(-1).tolist()

            if isinstance(value, (list, tuple)):
                if value and any(
                    isinstance(item, (torch.Tensor, list, tuple)) for item in value
                ):
                    return [item for item in value]
                return [int(item) for item in value]

            if hasattr(value, "tolist"):
                raw = value.tolist()
                if isinstance(raw, list):
                    return [int(item) for item in raw]
                return [int(raw)]

            return [int(value)]

        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            first = tensors[0]
            if first and any(
                isinstance(item, (torch.Tensor, list, tuple)) for item in first
            ):
                tensors = tuple(first)

        def decode_one(value):
            ids = to_ids(value)
            if ids and isinstance(ids[0], (torch.Tensor, list, tuple)):
                return [decode_one(item) for item in ids]
            return "".join([self.id_to_token[x] for x in ids]).replace("G̃", " ")

        texts = [decode_one(t) for t in tensors]

        return texts[0] if len(texts) == 1 else texts


if __name__ == "__main__":
    tokenizeri = BPETokenizer()

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
