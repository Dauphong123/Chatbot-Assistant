import os
import json

import tiktoken
import torch
from paths import CHECKPOINT_DIR, DATA_DIR, ROOT_DIR
from torch.utils.data import DataLoader
from training import generate
from scripts.paths import CHECKPOINT_DIR, DATA_DIR

from configs.model_config import GPT_CONFIG
from src.llm import GPTDataset, GPTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = tiktoken.get_encoding("gpt2")
PROMPT_FILE = os.path.join(ROOT_DIR, "evaluate_prompt.txt")
OUTPUT_FILE = os.path.join(CHECKPOINT_DIR, "evaluate_output.txt")
FINETUNING_PATH = os.path.join(CHECKPOINT_DIR, "finetuning", "best.pth")
PRETRAINING_PATH = os.path.join(CHECKPOINT_DIR, "pretraining", "best.pth")


def has_repetition(text, n=3):
    words = text.split()

    if len(words) < n * 2:
        return False

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    return len(ngrams) != len(set(ngrams))


def load_prompt(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_path, device):
    model = GPTModel(GPT_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, device, max_tokens=50):
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    token_ids = generate(
        model,
        token_ids,
        max_new_tokens=max_tokens,
        context_size=GPT_CONFIG["context_length"],
        temparature=0.2,
        top_k=50,
    )
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def evaluate_model(models, tokenizer, prompts, device):
    results = []

    repetition_count = 0
    for i, item in enumerate(prompts):
        input_result = {}
        prompt = item["prompt"]
        has_repetition_flag = False

        print(f"\n{item['id']}")
        print("prompt")

        output = []
        with torch.no_grad():
            for model in models:
                output.append(generate_text(model, tokenizer, prompt, device))

        print(f"Answer: {output}")

        if any(has_repetition(text) for text in output):
            repetition_count += 1

        input_result = {
            "category": item["theme"],
            "prompt": prompt,
            "has_repetition": has_repetition_flag,
        }

        for i in range(len(models)):
            input_result[f"model_{i + 1}_output"] = output[i]

        results.append(input_result)
        """
        theme = item["theme"]
        if theme not in category_results:
            category_results[theme] = [0] * len(models)
        """

    repetition_rate = repetition_count / len(prompts)
    print(f"Repetition rate: {repetition_rate:.2%}")

    return results


def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    pretrain_model = load_model(
        PRETRAINING_PATH,
        DEVICE,
    )

    finetune_model = load_model(FINETUNING_PATH, DEVICE)

    val_dataset = GPTDataset(
        os.path.join(DATA_DIR, "writing_val.bin"),
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    prompts = load_prompt(PROMPT_FILE)

    pretrain_results = evaluate_model(
        [pretrain_model, finetune_model], TOKENIZER, prompts, DEVICE
    )

    print(f"\nSaved results to {OUTPUT_FILE}")
