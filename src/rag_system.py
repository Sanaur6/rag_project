import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

from src.loader import chunk_text, load_documents


load_dotenv()


class RAGSystem:
    def __init__(
        self,
        docs_folder: str = "data/documents",
        top_k: int = 3,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        self.docs_folder = Path(docs_folder)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.embedder = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.vocab: List[str] = []
        self.vocab_index: Dict[str, int] = {}

        self.chunks: List[Dict[str, Any]] = self._load_chunks()
        self.embeddings = self._get_embeddings()
        self.index: Optional[faiss.Index] = self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", text.lower())

    def _build_vocabulary(self) -> None:
        tokens: set[str] = set()
        for chunk in self.chunks:
            tokens.update(self._tokenize(chunk.get("text", "")))

        self.vocab = sorted(tokens)
        self.vocab_index = {token: idx for idx, token in enumerate(self.vocab)}

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if not self.vocab:
            self._build_vocabulary()

        if not self.vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)

        vectors = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)

        for row_index, text in enumerate(texts):
            token_counts = Counter(self._tokenize(text))
            for token, count in token_counts.items():
                idx = self.vocab_index.get(token)
                if idx is not None:
                    vectors[row_index, idx] = float(count)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        return vectors.astype(np.float32)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        raw_documents = load_documents(str(self.docs_folder))
        chunks: List[Dict[str, Any]] = []

        for document in raw_documents:
            text = document.get("text", "")
            for chunk in chunk_text(text, self.chunk_size, self.overlap):
                chunks.append(
                    {
                        "text": chunk,
                        "source": document.get("source", "unknown"),
                        "page": document.get("page"),
                    }
                )

        return chunks

    def _get_embeddings(self) -> np.ndarray:
        if not self.chunks:
            return np.empty((0, 0), dtype=np.float32)

        self._build_vocabulary()
        texts = [chunk["text"] for chunk in self.chunks]
        return self._encode_texts(texts)

    def _build_index(self) -> Optional[faiss.Index]:
        if self.embeddings.size == 0:
            return None

        matrix = self.embeddings.copy()
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def query(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.chunks or self.index is None:
            return []

        limit = top_k if top_k is not None else self.top_k
        limit = max(1, min(limit, len(self.chunks)))

        query_vector = self._encode_texts([question])
        if query_vector.size == 0:
            return []
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, limit)

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.chunks):
                continue

            chunk = self.chunks[int(index)]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "score": float(score),
                }
            )

        return results

    def _fallback_answer(self, question: str, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return "I couldn't find that information in the available documents."

        keywords = [word.lower() for word in question.split() if len(word) > 3]

        for source in sources:
            text = source["text"].lower()
            if keywords and any(keyword in text for keyword in keywords):
                return source["text"]

        return "Based on the available documents: " + sources[0]["text"][:500]

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ) -> Any:
        sources = self.query(question, top_k=top_k)

        if not sources:
            response = "I couldn't find that information in the available documents."
            return (response, []) if return_sources else response

        context = "\n\n---\n\n".join(source["text"] for source in sources)

        if self.api_key:
            try:
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    "You are a helpful company policy assistant. "
                    "Answer using only the context provided below. "
                    "If the answer is not in the context, say exactly: 'I couldn't find that information in the available documents.'\n\n"
                    f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                answer = getattr(response, "text", str(response))
                return (answer, sources) if return_sources else answer
            except Exception:
                pass

        answer = self._fallback_answer(question, sources)
        return (answer, sources) if return_sources else answer


if __name__ == "__main__":
    rag = RAGSystem(docs_folder="data/documents")
    sample_question = "What are the password requirements?"
    print(rag.answer(sample_question))
