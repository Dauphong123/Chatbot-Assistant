from .components import (
    Embedder,
    Chunker,
    Vector_store,
    SearchEngineLoader,
    Reranker,
    Document,
)


class RAG:
    def __init__(self, threshhold=0.8):
        self.embedder = Embedder()
        self.chunker = Chunker()
        self.vector_store = Vector_store()
        self.search_engine = SearchEngineLoader()
        self.threshhold = threshhold
        self.encoder = Reranker()

    def rerank(self, query_pairs):
        scores = self.encoder.rerank(query_pairs)
        return scores

    def search(self, query, max_results=10, rerank=3):
        search_results = self.search_engine.search(query, max_results)

        for result in search_results:
            href = result["href"]
            body = result["text"]
            document = Document(body, metadata={"source": href})
            self.add_document(document)

    def add_document(self, document):
        chunks = self.chunker.recursive_chunk(document)

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed(texts)
        self.vector_store.add_document(chunks, embeddings)

    def retrieve(self, query, top_k=20, rerank=3):
        query_embedding = self.embedder.embed([query])[0]

        results = self.vector_store.search(
            query_embedding, top_k=top_k, threshhold=self.threshhold
        )

        if rerank > 0:
            pairs = [(query, chunk["document"].text) for chunk in results]
            scores = self.rerank(pairs)

            for score, result in zip(scores, results):
                result["rank_score"] = score

            ranked = sorted(
                results,
                key=lambda x: x["rank_score"],
                reverse=True,
            )
            return ranked[:rerank]
        else:
            return results
