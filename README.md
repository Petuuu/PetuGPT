# Building an LLM from scratch

## Overview

A GPT-2 model implemented, pre-trained and fine-tuned using *PyTorch*.


### Usage

You can run the model either locally or using the web interface at https://example.com

#### Local use

If you don't already have *PyTorch* and *tiktoken*, install it with `pip install torch tiktoken`.

Once you have *PyTorch* installed, navigate to the 'gpt' directory. Finally, run the model with `py -m run {FLAGS}`.

By default, the model loaded with OpenAI weights is used. You can use the following flags:
- -l/--local-pretrain: uses locally pre-trained model
- -c/--classification: spam classification (fine-tuned)
- -i/--instruction: follows user given instruction (fine-tuned)


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
Token embeddings      | vocab size × embedding dim                                    | 46,180,352 - 51,463,168 (OpenAI)
Positional embeddings | context length × embedding dim                                | 262,144 - 1,048,576 (OpenAI)
Multi-head attention  | 4 × embedding dim ^ 2 + embedding dim (Q, K, V, output)       | 4,195,328
Feed-forward          | 8 × embedding dim ^ 2 + 5 × embedding dim (bias)              | 8,393,728
Transformer blocks    | no. layers × (multi-head attention + feed-forward + \* )      | 302,235,648
Final normalization   | 2 × embedding dim                                             | 2,048
Output                | embedding dim × vocab size / 2 (spam classification 0/1)      | 46,180,352 (no weight tying with own) / 2
||
**Total**             | embeddings + transformer blocks + final norm (+ output layer) | 354,749,440(+2) (OpenAI) - 394,860,544(-46,180,350)

\* 2 × normalization layers = 4 × embedding dim = 4,096

### Datasets
For pretraining: allenai/peS20 from Hugging Face (https://huggingface.co/datasets/allenai/peS2o)
- ~30 MB used for creating BPE vocabulary
- ~450 MB used for pre-training
- ~50 MB used for validation

For classification fine-tuning: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

For instruction fine-tuning: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json

### Tokenizer
- Built from scratch: Byte-Pair Encoding (BPE)
- With preloaded OpenAI weights: tiktoken gpt2
