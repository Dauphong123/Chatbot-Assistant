GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

PRETRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 5000,
}

FINETUNING_CONFIG = {
    "learning_rate": 3e-5,
    "min_learning_rate": 3e-6,
    "warmup_steps": 100,
}

