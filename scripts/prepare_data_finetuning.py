import os

import tiktoken
from datasets import concatenate_datasets, load_dataset
from paths import DATA_DIR

from configs.model_config import GPT_CONFIG
import hashlib
import re

TRAIN_FILE = os.path.join(DATA_DIR, "SFT_train.parquet")
VAL_FILE = os.path.join(DATA_DIR, "SFT_val.parquet")

IGNORE_INDEX = -100
MAX_CONTEXT = GPT_CONFIG["context_length"]
MAX_TOKEN = 600_000

smoltalk_dataset = load_dataset(
    "HuggingFaceTB/smoltalk",
    "all",
)
ultra_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
alpaca_dataset = load_dataset("vicgalle/alpaca-gpt4")
svamp_dataset = load_dataset("ChilleD/SVAMP")

tokenizer = tiktoken.get_encoding("gpt2")


def ngram_repetition_ratio(text, n=0):
    words = text.lower().split()

    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    unique_ngrams = set(ngrams)

    return 1 - len(unique_ngrams) / len(ngrams)


def get_conversation_hash(messages):
    """
    Creates a unique hash based on user/assistant text content.
    """
    combined_text = "".join(
        [m["content"] for m in messages if m["role"] in ["user", "assistant"]]
    )
    return hashlib.md5(combined_text.encode("utf-8")).hexdigest()


def clean_content(content):
    """
    Remove EOS markers that may already exist in the dataset.
    We add exactly one EOS token ourselves to assistant responses.
    """
    return content.replace("<|endoftext|>", "").strip()


def format_message(message):
    """
    Format messages exactly the same way they will be used during inference.
    """

    role = message["role"]
    content = clean_content(message["content"])

    if role == "system":
        return ""
    if role == "user":
        return f"### User:\n{content}\n"

    if role == "assistant":
        return f"### Response:\n{content}"

    return ""


def get_token_count(conversation):
    """
    Count tokens using exactly the same formatting as process_conversation().
    """

    total = 0

    for message in conversation:
        role = message["role"]
        text = format_message(message)

        if not text:
            continue

        tokens = tokenizer.encode(text)

        # Assistant responses end with exactly one EOS token.
        if role == "assistant":
            tokens.append(tokenizer.eot_token)

        total += len(tokens)

    return total


def process_conversation(conversation):
    """
    Convert one conversation into input_ids and labels.

    System/user tokens:
        labels = -100

    Assistant tokens:
        labels = actual token IDs

    EOS is included in the assistant target.
    """

    input_ids = []
    labels = []

    system_message = (
        "### System:\nYou are a helpful, accurate, and concise AI assistant.\n"
    )
    tokens = tokenizer.encode(system_message, allowed_special={"<|endoftext|>"})
    input_ids.extend(tokens)
    labels.extend([IGNORE_INDEX] * len(tokens))

    for message in conversation:
        role = message["role"]
        text = format_message(message)

        if not text:
            continue

        tokens = tokenizer.encode(text)

        if role == "assistant":
            # Add exactly one EOS token.
            lenght_prefix = tokenizer.encode("### Response:\n")
            tokens.append(tokenizer.eot_token)

            labels_token = [IGNORE_INDEX] * len(lenght_prefix) + tokens[
                len(lenght_prefix) :
            ]
            input_ids.extend(tokens)
            labels.extend(labels_token)

        else:
            # System/user text is context, not training target.
            input_ids.extend(tokens)
            labels.extend([IGNORE_INDEX] * len(tokens))

    # Causal language-model shift.
    #
    # Input:
    #     A B C D
    #
    # Labels:
    #     B C D EOS
    #
    # The model predicts the next token.
    input_ids = input_ids[:-1]
    labels = labels[1:]

    return input_ids, labels


def process_batch(batch):
    batch_input = []
    batch_label = []

    for conversation in batch["messages"]:
        input_ids, labels = process_conversation(conversation)

        batch_input.append(input_ids)
        batch_label.append(labels)

    return {
        "input_ids": batch_input,
        "labels": batch_label,
    }


def alpaca_format(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, accurate, and concise AI assistant. Answer the user's questions clearly and directly.",
            },
            {
                "role": "user",
                "content": example["instruction"].strip()
                + "\n\n"
                + example["input"].strip(),
            },
            {"role": "assistant", "content": example["output"].strip()},
        ]
    }


def svamp_format(example):
    body = example["Body"].strip()
    question = example["Question"].strip()
    equation = example["Equation"].strip()
    result = str(example["Answer"]).strip()

    return {
        "messages": [
            {"role": "user", "content": body + ". " + question},
            {"role": "assistant", "content": equation + "=" + result},
        ]
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    smoltalk_train = smoltalk_dataset["train"]
    smoltalk_val = smoltalk_dataset["test"]

    ultra_train = ultra_dataset["train_sft"]
    ultra_val = ultra_dataset["test_sft"]

    svamp_train = svamp_dataset["train"]
    svamp_val = svamp_dataset["test"]

    svamp_train = svamp_train.map(svamp_format, remove_columns=svamp_train.column_names)
    svamp_val = svamp_val.map(svamp_format, remove_columns=svamp_val.column_names)

    alpaca_train = alpaca_dataset["train"]

    alpaca_train = alpaca_train.map(
        alpaca_format,
        remove_columns=alpaca_train.column_names,
    )
    print(alpaca_train[0])

    ds_train = concatenate_datasets(
        [smoltalk_train, ultra_train, alpaca_train, svamp_train]
    )
    ds_val = concatenate_datasets([smoltalk_val, ultra_val, svamp_val])

    print(f"Original train samples: {len(ds_train)}")
    print(f"Original val samples:   {len(ds_val)}")

    # ---------------------------------------------------------
    # Filter sequences that fit inside the model context.
    # ---------------------------------------------------------

    ds_train = ds_train.filter(
        lambda x: get_token_count(x["messages"]) <= MAX_CONTEXT,
        num_proc=4,
    )

    ds_val = ds_val.filter(
        lambda x: get_token_count(x["messages"]) <= MAX_CONTEXT,
        num_proc=4,
    )

    print(f"Train samples after filtering: {len(ds_train)}")
    print(f"Val samples after filtering:   {len(ds_val)}")

    # ---------------------------------------------------------
    # Shuffle training data.
    # ---------------------------------------------------------
    print("\nDeduplicating datasets...")

    # Train deduplication
    seen_train_hashes = set()

    def filter_train_duplicates(example):
        h = get_conversation_hash(example["messages"])
        if h in seen_train_hashes:
            return False
        seen_train_hashes.add(h)
        return True

    # Note: Keep num_proc=None (single-process) so the set correctly blocks global duplicates
    ds_train = ds_train.filter(filter_train_duplicates, num_proc=None)

    # Validation deduplication
    seen_val_hashes = set()

    def filter_val_duplicates(example):
        h = get_conversation_hash(example["messages"])
        if h in seen_val_hashes:
            return False
        seen_val_hashes.add(h)
        return True

    ds_val = ds_val.filter(filter_val_duplicates, num_proc=None)

    print(f"Train samples after deduplication: {len(ds_train)}")
    print(f"Val samples after deduplication:   {len(ds_val)}")

    def is_high_quality(messages):
        """
        Returns False if the assistant responses show signs of low quality.
        """
        for message in messages:
            if message["role"] != "assistant":
                continue

            content = message["content"].strip()

            # 2. Filter out raw code errors / empty markdown snippets
            if content in ["```", "```python\n```", "Undefined"]:
                return False

            # 3. Filter out repetitive generation loops (common LLM failure mode)
            # Check if a single phrase or token repeats endlessly
            words = content.split()
            if len(words) > 20:
                # If the top 3 most common words make up > 40% of the entire text,
                # it's usually a degenerated repetition loop (e.g., "the the the...")
                from collections import Counter

                word_counts = Counter(words)
                most_common_freq = sum(count for _, count in word_counts.most_common(3))
                if most_common_freq / len(words) > 0.40:
                    return False

            # 4. Filter out typical refusal boilerplate if you want a proactive model
            # Un-comment this if you want to explicitly avoid "As an AI language model..." responses
            # low_quality_phrases = ["as an ai language model", "i do not have personal opinions"]
            # if any(phrase in content.lower() for phrase in low_quality_phrases):
            #     return False

        return True

    print("\nFiltering out low-quality answers...")

    ds_train = ds_train.filter(lambda x: is_high_quality(x["messages"]), num_proc=4)
    ds_val = ds_val.filter(lambda x: is_high_quality(x["messages"]), num_proc=4)

    print("\nFiltering out lazy boilerplate...")

    LOW_QUALITY_BOILERPLATE = [
        "i'm sorry, but as an ai",
        "as a helpful assistant",
        "sure, i can help you with that",
        "great question!",
        "i'd be happy to help",
    ]

    def removes_lazy_boilerplate(messages):
        for m in messages:
            if m["role"] == "assistant":
                text_lower = m["content"].lower()
                # If the assistant begins its response with fluffy filler text, skip it
                if any(
                    text_lower.startswith(phrase) for phrase in LOW_QUALITY_BOILERPLATE
                ):
                    return False
        return True

    ds_train = ds_train.filter(
        lambda x: removes_lazy_boilerplate(x["messages"]), num_proc=4
    )
    ds_val = ds_val.filter(
        lambda x: removes_lazy_boilerplate(x["messages"]), num_proc=4
    )

    def filter_repetitive(example):
        for message in example["messages"]:
            if message["role"] != "assistant":
                continue

            text = message["content"]

            repetition = ngram_repetition_ratio(text, n=3)

            if repetition > 0.30:
                return False

        return True

    ds_train = ds_train.filter(filter_repetitive)
    ds_val = ds_val.filter(filter_repetitive)

    print(f"Train samples after quality filter: {len(ds_train)}")
    print(f"Val samples after quality filter:   {len(ds_val)}")

    ds_train = ds_train.shuffle(seed=42)

    # Limit training set.
    ds_train = ds_train.select(range(min(MAX_TOKEN, len(ds_train))))

    print(f"Train samples selected: {len(ds_train)}")

    # ---------------------------------------------------------
    # Tokenize.
    # ---------------------------------------------------------

    tokenized_train_ds = ds_train.map(
        process_batch,
        batched=True,
        num_proc=4,
        remove_columns=ds_train.column_names,
    )

    tokenized_val_ds = ds_val.map(
        process_batch,
        batched=True,
        num_proc=4,
        remove_columns=ds_val.column_names,
    )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    tokenized_train_ds.to_parquet(TRAIN_FILE)

    tokenized_val_ds.to_parquet(VAL_FILE)

    print()
    print("===================================")
    print("SFT dataset successfully created")
    print("===================================")

    print(f"Train file: {TRAIN_FILE}")
    print(f"Val file:   {VAL_FILE}")

    # ---------------------------------------------------------
    # Inspect one training sample.
    # ---------------------------------------------------------

    sample = tokenized_train_ds[0]

    print()
    print("========== SAMPLE ==========")

    print()
    print("INPUT:")
    print(tokenizer.decode(sample["input_ids"]))

    valid_labels = [token for token in sample["labels"] if token != IGNORE_INDEX]

    print()
    print("TARGET:")
    print(tokenizer.decode(valid_labels))

    print()
    print("Input length:", len(sample["input_ids"]))
    print("Label length:", len(sample["labels"]))

    # ---------------------------------------------------------
    # Verify lengths are identical.
    # ---------------------------------------------------------

    assert len(sample["input_ids"]) == len(sample["labels"])

    print()
    print("Input/label lengths match.")

    # ---------------------------------------------------------
    # Verify EOS exists in assistant target.
    # ---------------------------------------------------------

    if valid_labels:
        print("Last target token:", valid_labels[-1])

        print("EOS token:", tokenizer.eot_token)

        if valid_labels[-1] == tokenizer.eot_token:
            print("EOS check: OK")
        else:
            print("WARNING: EOS is missing!")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
