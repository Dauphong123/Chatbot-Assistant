import os

import numpy as np
import tiktoken
from datasets import load_dataset
from paths import DATA_DIR

TRAIN_FILE = os.path.join(DATA_DIR, "test_train.bin")
VAL_FILE = os.path.join(DATA_DIR, "test_val.bin")

MAX_TOKEN = 50_000

VAL_RATIO = 0.1
TRAIN_RATIO = 1 - VAL_RATIO

PRINT_EVERY = 10

DTYPE = np.uint16

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
)


def main(dataset):
    os.makedirs(DATA_DIR, exist_ok=True)
    tokenizer = tiktoken.get_encoding("gpt2")
    if not tokenizer:
        print("tokenizer not loaded")
        return

    with open(TRAIN_FILE, "wb") as train_file, open(VAL_FILE, "wb") as val_file:
        total_token = 0
        train_tokens, val_tokens = 0, 0
        train_token_target = int(MAX_TOKEN * TRAIN_RATIO)
        val_token_target = MAX_TOKEN - train_token_target
        for i, example in enumerate(dataset):
            text = example["text"]

            tokens = tokenizer.encode_ordinary(text)
            tokens.append(tokenizer.eot_token)
            tokens = np.asarray(tokens, dtype=np.uint16)
            total_token += len(tokens)

            if train_tokens < train_token_target:
                remaining = train_token_target - train_tokens

                train_chunk = tokens[:remaining]

                train_file.write(train_chunk.tobytes())

                train_tokens += len(train_chunk)

                tokens = tokens[len(train_chunk) :]

            if len(tokens) > 0 and val_tokens < val_token_target:
                remaining = val_token_target - val_tokens

                val_chunk = tokens[:remaining]

                val_file.write(val_chunk.tobytes())

                val_tokens += len(val_chunk)

            if i % PRINT_EVERY == 0:
                print("Document", i)
                print("Train: ", train_tokens)
                print("Val: ", val_tokens)

            if train_tokens >= train_token_target and val_tokens >= val_token_target:
                break

    print("get data end. get maximum token")


if __name__ == "__main__":
    main(dataset=dataset)
