import tiktoken # byte-pair tokenizer

with open("the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("o200k_base")

sample = tokenizer[:50]







