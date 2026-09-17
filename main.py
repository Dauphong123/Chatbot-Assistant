import os
from pathlib import Path
from collections import deque

import tiktoken
import torch

from configs.model_config import GPT_CONFIG
from scripts.paths import CHECKPOINT_DIR, DOCUMENT_DIR
from scripts.training import token_ids_to_text
from src.context_build import ContextBuilder
from src.llm import GPTModel
from src.rag.components import Document
from src.rag.rag import RAG


MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "finetuning",
    "best.pth",
)

tokenizer = tiktoken.get_encoding("gpt2")


# ============================================================
# Generation settings
# ============================================================

REPETITION_PENALTY = 1.05
TEMPERATURE = 0.0
TOP_K = None

# Reserve room for the answer.
MAX_NEW_TOKENS = 96


# ============================================================
# Conversation settings
# ============================================================

# Maximum number of messages stored in memory.
#
# The prompt builder dynamically decides how many of these
# can fit into the model context.
MAX_HISTORY_MESSAGES = 50


# ============================================================
# Repetition penalty
# ============================================================


def apply_repetition_penalty(
    logits,
    idx,
    penalty=1.05,
):
    for token_id in set(idx[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits


# ============================================================
# Repeated n-gram detection
# ============================================================


def has_repeated_ngram(
    tokens,
    n=3,
    max_repeats=2,
):
    """
    Stop pathological repetition such as:

        correctly correctly correctly
        k = 4 / 2
        k = 4 / 2
        k = 4 / 2
    """

    if len(tokens) < n * (max_repeats + 1):
        return False

    recent = tokens[-n:]

    count = 0

    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == recent:
            count += 1

    return count > max_repeats


# ============================================================
# Generation
# ============================================================


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Last token
        logits = logits[:, -1, :]

        # Repetition penalty
        logits = apply_repetition_penalty(
            logits,
            idx,
            REPETITION_PENALTY,
        )

        # Top-k
        if top_k is not None:
            top_logits, _ = torch.topk(
                logits,
                top_k,
            )

            min_val = top_logits[:, -1].unsqueeze(-1)

            logits = torch.where(
                logits < min_val,
                torch.full_like(
                    logits,
                    float("-inf"),
                ),
                logits,
            )

        # Sampling
        if temperature > 0:
            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1,
            )

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

        idx = torch.cat(
            (idx, idx_next),
            dim=1,
        )

    return idx


# ============================================================
# RAG retrieval
# ============================================================


def context_query(
    rag,
    query_text,
    context_builder,
    top_k=20,
    max_results=50,
    reranking=3,
):
    results = rag.retrieve(
        query_text,
        top_k,
        reranking,
    )

    if not results:
        rag.search(
            query_text,
            max_results=max_results,
        )

        results = rag.retrieve(
            query_text,
            top_k,
            reranking,
        )

    if not results:
        return ""

    context = context_builder.build(results)

    return context.strip()


# ============================================================
# Load documents
# ============================================================


def load_rag_documents(
    rag,
    docs_dir,
):
    for path in Path(docs_dir).glob("*.txt"):
        try:
            text = path.read_text(
                encoding="utf-8",
            )

        except UnicodeDecodeError:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        if text.strip():
            rag.add_document(
                Document(
                    text,
                    {
                        "source": str(path),
                    },
                )
            )

    return rag


# ============================================================
# Conversation queue
# ============================================================


def add_message(
    conversation,
    role,
    content,
):
    conversation.append(
        {
            "role": role,
            "content": content,
        }
    )


# ============================================================
# Token counting
# ============================================================


def count_tokens(text):
    return len(
        tokenizer.encode(
            text,
            allowed_special={
                "<|endoftext|>",
            },
        )
    )


# ============================================================
# Conversation formatting
# ============================================================


def format_message(message):
    if message["role"] == "user":
        return "### User:\n" + message["content"]

    return "### Response:\n" + message["content"]


# ============================================================
# Build prompt
# ============================================================


def build_prompt(
    user_input,
    context,
    conversation,
):
    """
    Build:

        ### System:
        system instructions

        ### User:
        previous question

        ### Response:
        previous answer

        ### User:
        previous question

        ### Response:
        previous answer

        ### System:
        current instructions

        Retrieved context:
        ...

        ### User:
        current question

        ### Response:

    RAG receives only the current question.
    The LLM receives recent conversation history.

    The oldest history is removed first when the prompt
    reaches the token budget.
    """

    context_length = GPT_CONFIG["context_length"]

    # Reserve space for generated answer.
    max_prompt_tokens = context_length - MAX_NEW_TOKENS

    # --------------------------------------------------------
    # System instruction
    # --------------------------------------------------------

    system_text = (
        "You are a strict factual assistant. "
        "Use the retrieved context when it is relevant. "
        "Use the previous conversation to understand the current question. "
        "Answer the current user question directly. "
        "Do not invent facts. "
        "Keep answers concise."
    )

    first_system = "### System:\n" + system_text

    # --------------------------------------------------------
    # Current system + retrieved context
    # --------------------------------------------------------

    current_system = (
        "\n\n### System:\n"
        "Use the retrieved context to answer the current user question."
    )

    if context:
        current_system += "\n\nRetrieved context:\n" + context

    # --------------------------------------------------------
    # Current user turn
    # --------------------------------------------------------

    current_turn = "\n\n### User:\n" + user_input + "\n### Response:\n"

    # --------------------------------------------------------
    # Select as much recent history as fits
    # --------------------------------------------------------

    selected_history = []

    messages = list(conversation)[-MAX_HISTORY_MESSAGES:]

    for message in reversed(messages):
        candidate_history = [message] + selected_history

        history_text = "\n\n".join(format_message(item) for item in candidate_history)

        candidate_prompt = (
            first_system + "\n\n" + history_text + current_system + current_turn
        )

        token_count = count_tokens(candidate_prompt)

        if token_count > max_prompt_tokens:
            break

        selected_history = candidate_history

    # --------------------------------------------------------
    # Build final prompt
    # --------------------------------------------------------

    prompt_parts = [
        first_system,
    ]

    for message in selected_history:
        prompt_parts.append("\n\n" + format_message(message))

    prompt_parts.append(current_system)

    prompt_parts.append(current_turn)

    return "".join(prompt_parts)


# ============================================================
# Load model and RAG
# ============================================================


def load_model_and_rag():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG)

    model.to(device)

    rag = RAG(
        threshhold=0.5,
    )

    rag = load_rag_documents(
        rag,
        DOCUMENT_DIR,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    model.eval()

    return model, rag, device


# ============================================================
# Generate response
# ============================================================


def generate_answer(
    model,
    prompt,
    device,
):
    input_ids = tokenizer.encode(
        prompt,
        allowed_special={
            "<|endoftext|>",
        },
    )

    context_length = GPT_CONFIG["context_length"]

    max_input_tokens = context_length - MAX_NEW_TOKENS

    # Safety check.
    #
    # build_prompt() should already keep the prompt
    # within this size.
    #
    # Keep the beginning so the system prompt is preserved.
    if len(input_ids) > max_input_tokens:
        input_ids = input_ids[:max_input_tokens]

    idx = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    response_start = idx.shape[1]

    generated_tokens = []

    # --------------------------------------------------------
    # Tokenize chat markers
    # --------------------------------------------------------

    user_marker = tokenizer.encode(
        "### User:",
        disallowed_special=(),
    )

    response_marker = tokenizer.encode(
        "### Response:",
        disallowed_special=(),
    )

    # ========================================================
    # Generate one token at a time
    # ========================================================

    while True:
        if idx.shape[1] >= context_length:
            break

        if len(generated_tokens) >= MAX_NEW_TOKENS:
            break

        old_length = idx.shape[1]

        idx = generate(
            model,
            idx,
            max_new_tokens=1,
            context_size=context_length,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )

        if idx.shape[1] == old_length:
            break

        next_token = idx[
            0,
            -1,
        ].item()

        # EOS
        if next_token == tokenizer.eot_token:
            break

        generated_tokens.append(next_token)

        # ----------------------------------------------------
        # Stop if model starts another user turn
        # ----------------------------------------------------

        if len(generated_tokens) >= len(user_marker):
            if generated_tokens[-len(user_marker) :] == user_marker:
                break

        # ----------------------------------------------------
        # Stop if model starts another response section
        # ----------------------------------------------------

        if len(generated_tokens) >= len(response_marker):
            if generated_tokens[-len(response_marker) :] == response_marker:
                break

        # ----------------------------------------------------
        # Stop pathological repetition
        # ----------------------------------------------------

        if has_repeated_ngram(
            generated_tokens,
            n=3,
            max_repeats=2,
        ):
            break

    # ========================================================
    # Extract generated tokens
    # ========================================================

    output_ids = idx[
        :,
        response_start:,
    ]

    text = token_ids_to_text(
        tokenizer,
        output_ids,
    )

    # Remove accidentally generated chat markers.
    text = text.replace(
        "### User:",
        "",
    )

    text = text.replace(
        "### Response:",
        "",
    )

    text = text.replace(
        "<|endoftext|>",
        "",
    )

    return text.strip()


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 60)
    print("Loading model and RAG...")
    print("=" * 60)

    model, rag, device = load_model_and_rag()

    context_builder = ContextBuilder()

    # --------------------------------------------------------
    # Conversation queue
    # --------------------------------------------------------

    conversation = deque(maxlen=MAX_HISTORY_MESSAGES)

    print("=" * 60)
    print("RAG chatbot ready.")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nUser: ")

        if user_input.strip().lower() == "quit":
            break

        # ----------------------------------------------------
        # RAG ONLY gets the current user query.
        # ----------------------------------------------------

        retrieval_query = user_input.strip()

        print("\n[Retrieval query: " + retrieval_query + "]")

        context = context_query(
            rag,
            retrieval_query,
            context_builder,
            top_k=5,
            max_results=20,
            reranking=3,
        )

        # ----------------------------------------------------
        # Build LLM prompt.
        #
        # The LLM gets:
        #
        #   previous conversation
        #   current retrieved context
        #   current question
        # ----------------------------------------------------

        prompt = build_prompt(
            user_input=user_input,
            context=context,
            conversation=conversation,
        )

        # ----------------------------------------------------
        # Optional debugging:
        #
        # print("\n" + prompt + "\n")
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Store current user message
        # ----------------------------------------------------

        add_message(
            conversation,
            "user",
            user_input,
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        text = generate_answer(
            model=model,
            prompt=prompt,
            device=device,
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print(
            "Assistant:",
            text,
        )

        # ----------------------------------------------------
        # Store assistant response
        # ----------------------------------------------------

        add_message(
            conversation,
            "assistant",
            text,
        )


if __name__ == "__main__":
    main()
