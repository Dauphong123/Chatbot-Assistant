import numpy as np


class Vector_store:
    def __init__(self):
        self.documents = []
        self.embedding = []

    def add_document(self, document, embedding):
        self.documents.extend(document)
        self.embedding.extend(embedding)

    def search(self, query_embedding, top_k=5, threshhold=0.5):
        if not self.embedding:
            return []

        embedding_array = np.array(self.embedding)

        scores = embedding_array @ query_embedding

        indices = np.argsort(scores)[::-1]

        results = []

        for idx in indices:
            score = scores[idx]

            if score < threshhold:
                break

            results.append(
                {
                    "document": self.documents[idx],
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results
