# neurolens_rag.py

from sentence_transformers import SentenceTransformer
import numpy as np

class NeuroLensRAG:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = []

    # -------------------------------
    # 1. Split text into chunks
    # -------------------------------
    def chunk_text(self, text, chunk_size=2):
        sentences = text.split('.')
        chunks = []

        for i in range(0, len(sentences), chunk_size):
            chunk = '.'.join(sentences[i:i+chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    # -------------------------------
    # 2. Store embeddings
    # -------------------------------
    def build_index(self, text):
        self.chunks = self.chunk_text(text)
        self.embeddings = self.model.encode(self.chunks)

    # -------------------------------
    # 3. Retrieve relevant chunks
    # -------------------------------
    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query])[0]

        similarities = []

        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            similarities.append(sim)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [self.chunks[i] for i in top_indices]

        return " ".join(results)