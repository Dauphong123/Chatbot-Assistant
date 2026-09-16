import os
from pathlib import Path

import torch

from configs.model_config import GPT_CONFIG
from scripts.paths import CHECKPOINT_DIR, DOCUMENT_DIR
from src.llm.model import GPTModel
from src.rag import RAG, Document

model_path = os.path.join(CHECKPOINT_DIR, "finetuning", "best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG)
model.to(DEVICE)

rag = RAG(model, threshhold=0.5)
for path in Path(os.path.join(DOCUMENT_DIR)).glob("*.txt"):
    text = path.read_text(encoding="utf-8")

    rag.add_document(Document(text, {"source": str(path)}))

query = "what is sql?"
rag.search(query, max_results=10)

results = rag.retrieve(query, top_k=3)
for result in results:
    doc = result["document"]

    print("=" * 60)
    print("Score:", result["score"])
    print("Source:", doc.metadata["source"])
    print()
    print(doc.text)
