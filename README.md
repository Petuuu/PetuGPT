# Building an LLM from scratch

## Overview

A GPT-2 model implemented, pre-trained and fine-tuned using *PyTorch*.


### Configuration and parameters
.                   | Count
---                 | ---
Vocabulary size     | 45,098 with own model, 50,257 with OpenAI weights
Context length      | 256 with own model, 1,024 with OpenAI weights
Embedding dimension | 1,024
No. heads           | 16
No. layers          | 24
Dropout rate        | 0.1

Layer                 | Origin                                                        | Params
---                   | ---                                                           | ---
Token embeddings      | vocab size × embedding dim                                    | 46,180,352 - 51,463,168
Positional embeddings | context length × embedding dim                                | 262,144 - 1,048,576
Multi-head attention  | 4 × embedding dim ^ 2 + embedding dim (Q, K, V, output)       | 4,195,328
Feed-forward          | 8 × embedding dim ^ 2 + 5 × embedding dim (bias)              | 8,393,728
Transformer blocks    | no. layers × (multi-head attention + feed-forward + \* )      | 302,235,648
Final normalization   | 2 × embedding dim                                             | 2,048
Output                | embedding dim × vocab size                                    | 46,180,352 (no weight tying with own)
||
**Total**             | embeddings + transformer blocks + final norm (+ output layer) | 354,749,440 (OpenAI) - 394,860,544

\* 2 × normalization layers = 4 × embedding dim = 4,096

### Datasets
allenai/peS20 from Hugging Face: https://huggingface.co/datasets/allenai/peS2o
- ~30 MB used for creating BPE vocabulary
- ~450 MB used for pre-training
- ~50 MB used for validation

### Tokenizer
- Built from scratch: Byte-Pair Encoding (BPE)
- With preloaded OpenAI weights: tiktoken gpt2

### NOTE
- Scripts written for Windows -> file paths need to be modified if run on different OS. Notebook, however, written for Unix
