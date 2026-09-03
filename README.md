# MLL from Scratch

A small GPT-2-style language model implemented with PyTorch, following the
ideas and progression in Sebastian Raschka's *Build a Large Language Model
(From Scratch)*.

The project trains a decoder-only transformer on the
`NahedAbdelgaber/evaluating-student-writing` dataset. It includes the model,
causal multi-head attention, feed-forward layers, normalization, tokenization,
binary dataset creation, and a pretraining loop.

## Project Layout

```text
.
├── pyproject.toml       # Package metadata and editable-install config
├── scripts/
│   ├── prepare_data.py  # Stream, tokenize, and save the training data
│   ├── pretrain.py      # Build the model and run pretraining
│   ├── training.py      # Training, evaluation, generation, and plotting
│   └── finetune.py      # Reserved for future fine-tuning code
├── configs/
│   └── model_config.py  # Model hyperparameters
├── requirements.txt     # Python dependencies recorded for the environment
├── data/
│   ├── writing_train.bin
│   └── writing_val.bin
├── checkpoints/         # Generated model checkpoints
├── plots/               # Generated loss plots
├── tests/               # Automated tests
└── src/llm/
    ├── dataset.py       # Memory-mapped token dataset
    ├── model.py         # GPTModel and TransformerBlock
    └── components/      # Attention, feed-forward, GELU, and normalization
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.x
- CUDA is recommended for the current training script

The scripts also import packages that are not currently listed in
`requirements.txt`. Install them explicitly if needed:

```bash
pip install -r requirements.txt
pip install datasets tiktoken
pip install -e .
```


## Quick Start

Run commands from the repository root so the relative `./data` paths resolve
correctly.

### 1. Create the token files

```bash
python scripts/prepare_data.py
```

This streams the `train` split from Hugging Face, encodes each document with
the GPT-2 tokenizer, and writes up to 800,000 `uint16` tokens split into:

- `data/writing_train.bin` (99%)
- `data/writing_val.bin` (1%)

The Hugging Face dataset may require internet access the first time it is
used.

### 2. Pretrain the model

```bash
python scripts/pretrain.py
```

The default configuration is:

| Setting | Value |
| --- | ---: |
| Vocabulary size | 50,257 |
| Context length | 256 |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Dropout | 0.1 |
| Epochs | 10 |
| Batch size | 2 |
| Gradient accumulation | 8 steps |

Training evaluates every 1,000 steps and prints training and validation loss.
Loss plots are displayed when training finishes.

## Generated Files

Training writes these artifacts to `checkpoints/`:

- `checkpoints/best_model.pth`: best checkpoint by validation loss, including
  model and optimizer state
- `checkpoints/gpt124m_final.pt`: final model state dictionary

The repository may also contain older checkpoints; they are not loaded
automatically by the current scripts.

## Model

`GPTModel` uses token and positional embeddings followed by 12 transformer
blocks. Each block applies pre-normalization, causal multi-head self-attention,
a feed-forward network, dropout, and residual connections. The output head
projects hidden states to GPT-2 vocabulary logits.

`GPTDataset` memory-maps the binary token files and returns next-token
prediction pairs. The current pretraining script uses non-overlapping chunks
of 256 tokens.

## Current Limitations

- `scripts/finetune.py` is currently empty.
- The training loop uses CUDA automatic mixed precision and a CUDA grad scaler;
	CPU-only training may require adapting those calls.
- Run `pip install -e .` before training so the `src/llm` package is importable.
- There are no automated tests or evaluation scripts yet.

## Reference

This project is a learning implementation inspired by:

> Sebastian Raschka, *Build a Large Language Model (From Scratch)*.
