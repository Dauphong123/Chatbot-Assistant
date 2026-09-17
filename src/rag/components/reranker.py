from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.encoder = CrossEncoder(str, cache_folder="./.cache")

    def rerank(self, pairs):
        return self.encoder.predict(pairs)
