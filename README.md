# MLL from Scratch

A small GPT-2-style decoder-only language model implemented with PyTorch. The
project follows Sebastian Raschka's _Build a Large Language Model (From
Scratch)_ and includes tokenization, binary datasets, causal attention,
feed-forward layers, normalization, pretraining, and text generation.

The pretraining pipeline currently streams the
`HuggingFaceFW/fineweb-edu` `sample-10BT` split.

## Project Layout

```text
.
├── pyproject.toml       # Package metadata and editable-install config
├── scripts/
│   ├── prepare_data_pretraining.py # Create binary pretraining data
│   ├── prepare_data_finetuning.py  # Create fine-tuning data
│   ├── pretrain.py                 # Run pretraining
│   ├── finetuning.py               # Run supervised fine-tuning
│   ├── evaluate.py                 # Evaluate and generate text
│   └── training.py                 # Shared training utilities
├── configs/model_config.py          # Model and optimizer configuration
├── requirements.txt                 # Python dependencies
├── data/                            # Generated datasets
├── checkpoints/                     # Generated model checkpoints
├── plots/                           # Generated loss plots
├── runs/                            # TensorBoard event files
└── src/llm/
    ├── dataset.py       # Memory-mapped token dataset
    ├── model.py         # GPTModel and TransformerBlock
    └── components/      # Attention, feed-forward, GELU, and normalization
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.x
- Internet access is required for the first Hugging Face dataset download
- CUDA is recommended for training

From the repository root, install the project with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data Preparation

Run commands from the repository root so the project paths resolve correctly.

```bash
python scripts/prepare_data_pretraining.py
```

This streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), encodes documents
with the GPT-2 tokenizer, and writes up to 50,000 `uint16` tokens using a
90/10 train/validation split:

```text
data/test_train.bin
data/test_val.bin
```

`scripts/pretrain.py` currently expects `data/writing_train.bin` and
`data/writing_val.bin`. Either copy the generated files before training, or
update the filenames used by the scripts:

```powershell
Copy-Item data/test_train.bin data/writing_train.bin
Copy-Item data/test_val.bin data/writing_val.bin
```

## Training

```bash
python scripts/pretrain.py
```

Pretraining saves checkpoints under `checkpoints/pretraining/`, the final
model state under `checkpoints/gpt124m_final.pt`, TensorBoard logs under
`runs/`, and loss plots under `plots/`.

View the logs with:

```bash
tensorboard --logdir runs
```

## Evaluation and Generation

After a compatible pretraining checkpoint exists, run:

```bash
python scripts/evaluate.py
```

The script loads `checkpoints/pretraining/best.pth`, reports validation loss
and perplexity, and generates text from the prompt `A student should`.

## Fine-Tuning

Fine-tuning expects:

```text
data/SFT_train.parquet
```

Run:

```bash
python scripts/prepare_data_finetuning.py
python scripts/finetuning.py
```

Fine-tuning loads `checkpoints/pretraining/best.pth` and writes checkpoints to
`checkpoints/finetuning/`.

## Model

`GPTModel` uses token and positional embeddings followed by 12 transformer
blocks. Each block applies pre-normalization, causal multi-head self-attention,
a feed-forward network, dropout, and residual connections. The output head
projects hidden states to GPT-2 vocabulary logits.

`GPTDataset` memory-maps the binary token files and returns next-token
prediction pairs. The current pretraining script uses non-overlapping chunks
of 256 tokens.

## Known Issues

- `scripts/pretrain.py` references an undefined `checkpoint_path` variable
  when creating the pretraining checkpoint directory.
- The data preparation and pretraining scripts use different binary filenames,
  as described above.
- Fine-tuning contains unfinished arguments and depends on the schema produced
  by `prepare_data_finetuning.py`.

## Reference

This project is a learning implementation inspired by:

> Sebastian Raschka, _Build a Large Language Model (From Scratch)_.

```


```
