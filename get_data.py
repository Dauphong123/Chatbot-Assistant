from datasets import load_dataset
import tiktoken
import numpy as np

DATASET_NAME = "sample-10BT"

TRAIN_FILE = "./data/fineweb_train.bin"
VAL_FILE = "./data/fineweb_val.bin"

MAX_TOKEN = 10_000_000

VAL_RATIO = 0.01
TRAIN_RATIO = 1 - VAL_RATIO

PRINT_EVERY = 10

DTYPE = np.uint16

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name=DATASET_NAME,
    split="train",
    streaming=True
)

def main(dataset, max_token = 10_000_000):
    tokenizer = tiktoken.get_encoding("gpt2")
    if not tokenizer: 
        print("tokenizer not loaded")
        return 

    train_file = open(TRAIN_FILE, "wb")
    if not train_file:
        print("no train file loaded")
        return
    val_file = open(VAL_FILE, "wb")
    if not val_file:
        print("no val file loaded")
        return

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

            tokens = tokens[len(train_chunk):]

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
    
        
    train_file.close()
    val_file.close()
    
    print("get data end. get maximum token")


if __name__ == "__main__":
    main(dataset=dataset)
