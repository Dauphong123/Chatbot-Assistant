from src.rag.document import Document

from .chunk import Chunker
from .embedding import Embedder
from .search_engine_loader import SearchEngineLoader
from .vector_store import Vector_store


class RAG:
    def __init__(self, llm, threshhold=0.8):
        self.embedder = Embedder()
        self.chunker = Chunker()
        self.vector_store = Vector_store()
        self.llm = llm
        self.search_engine = SearchEngineLoader()
        self.threshhold = threshhold

    def search(self, query, max_results=10):
        search_results = self.search_engine.search(query, max_results)

        for result in search_results:
            href = result["href"]
            body = result["body"]
            document = Document(body, metadata={"source": href})
            self.add_document(document)

    def add_document(self, document):
        chunks = self.chunker.recursive_chunk(document)

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.embed(texts)

        self.vector_store.add_document(chunks, embeddings)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.embed([query])[0]

        results = self.vector_store.search(
            query_embedding, top_k=top_k, threshhold=self.threshhold
        )

        return results
