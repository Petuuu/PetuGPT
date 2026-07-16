# Building an LLM from scratch

## Overview

A GPT-2 model implemented, pre-trained and fine-tuned using *PyTorch*.


### Configuration and parameters
.                   | Count
---                 | ---
Vocabulary size     | 38,338
Context length      | 4,096 -> 516 for training, 1,024 for pretrained weights
Embedding dimension | 1,024
No. heads           | 16
No. layers          | 24
Dropout rate        | 0.1

Layer                 | Origin                                                      | Params
---                   | ---                                                         | ---
Token embeddings      | vocab size × embedding dim                                  | 39,258,112
Positional embeddings | context length × embedding dim                              | TBD
Multi-head attention  | 4 × embedding dim ^ 2 + embedding dim (Q, K, V, output)     | 4,195,328
Feed-forward          | 8 × embedding dim ^ 2 + 5 × embedding dim (bias)            | 8,393,728
Transformer blocks    | no. layers × (multi-head attention + feed-forward + \* )    | 302,235,648
Final normalization   | 2 × embedding dim                                           | 2,048
Output                | embedding dim × vocab size                                  | 39,258,112 (no weight tying)
|
**Total**             | embeddings + transformer blocks + final norm + output layer | 380,753,920 + pos embeddings

\* 2 × normalization layers = 4 × embedding dim = 4,096

### Datasets
allenai/peS20 from Hugging Face: https://huggingface.co/datasets/allenai/peS2o
- 500 MB used for creating BPE vocabulary
- n GB used for pre-training
- n GB used for validation

### Tokenizer
- Byte-Pair Encoding (BPE)

### NOTE
- Scripts written for Windows -> file paths need to be modified if run on different OS. Notebook, however, written for Unix
