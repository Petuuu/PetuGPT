# Building an LLM from scratch

## Overview

A GPT-2 model implemented, pre-trained and fine-tuned using *PyTorch*.


### Configuration and parameters
-                   | -
---                 | ---
Vocabulary size     | 38,338
Context length      | 1,024
Embedding dimension | 768
No. heads           | 12
No. layers          | 12
Dropout rate        | 0.1

Layer                       | Origin                                                     | Params
---                         | ---                                                        | ---
Token embeddings            | vocab size × embedding dim                                 | 29,443,584
Positional embeddings       | context length × embedding dim                             | 786,432
Multi-head attention        | 4 × embedding dim ^ 2 (Q, K, V, output)                    | 2,359,296
Multilayer perceptron (MLP) | ???                                                        | ??? ???
Transformer blocks          | no. layers × (multi-head attention + MLP)                  | >28,311,552
Final LayerNorm             | embedding dim × ???                                        | ?? ???
Output                      | embedding dim × vocab size                                 | 29,443,584
|
**Total**                   | embeddings + transformer blocks + final LayerNorm + output | >90,344,448

### Datasets
allenai/peS20 from Hugging Face: https://huggingface.co/datasets/allenai/peS2o
- 500 MB used for creating BPE vocabulary
- n GB used for pre-training
- n GB used for validation

### Tokenizer
- Byte-Pair Encoding (BPE)