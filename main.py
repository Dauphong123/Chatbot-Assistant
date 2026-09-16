import os

import tiktoken
import torch

from configs.model_config import GPT_CONFIG
from scripts.paths import CHECKPOINT_DIR
from scripts.training import token_ids_to_text
from src.llm import GPTModel
from src.rag.rag import RAG

MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "finetuning",
    "best.pth",
)

tokenizer = tiktoken.get_encoding("gpt2")


def apply_repetition_penalty(logits, idx, penalty=1.2):
    for token_id in set(idx[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
):
    for _ in range(max_new_tokens):
        # Only use the last context_size tokens as model input
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Get logits for the last token
        logits = logits[:, -1, :]
        logits = apply_repetition_penalty(logits, idx, 1)

        # Top-k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)

            min_val = top_logits[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Choose next token
        if temperature > 0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )
        else:
            idx_next = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        # Append new token to the sequence
        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx


def query(rag, query_text, top_k=5):
    pass


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG)
    model.to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    conversation = []

    conversation.append("### System:\nYou are a helpful assistant.\n")

    while (user_input := input("User: ")) != "quit":
        # Build the new user turn
        conversation.append(f"### User:\n{user_input}\n### Response:\n")

        # Build complete prompt
        prompt = "".join(conversation)

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})

        idx = torch.tensor(input_ids).unsqueeze(0).to(device)

        # Remember where the assistant response starts
        response_start = idx.shape[1]

        # Generate one token at a time
        while True:
            idx = generate(
                model,
                idx,
                max_new_tokens=1,
                context_size=GPT_CONFIG["context_length"],
                temperature=0.7,
                top_k=50,
            )

            next_token = idx[0, -1].item()

            # GPT-2 EOS token
            if next_token == 50256:
                break

        # Extract only newly generated tokens
        output_ids = idx[:, response_start:]

        # Convert tokens back to text
        text = token_ids_to_text(tokenizer, output_ids)

        print("Assistant:", text.replace("<|endoftext|>", "").strip())

        # Add generated response to the last conversation turn
        conversation[-1] += text

        # Optional: prevent conversation from becoming too large
        total_tokens = idx.shape[1]

        if total_tokens > GPT_CONFIG["context_length"]:
            conversation = conversation[-1:]


if __name__ == "__main__":
    main()
