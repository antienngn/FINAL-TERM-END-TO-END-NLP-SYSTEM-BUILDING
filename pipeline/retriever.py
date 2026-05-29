from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .embedder import Embedder

DEFAULT_DB_PATH = "indexing/chroma_db"
DEFAULT_COLLECTION = "vnu_uet_rag"


@dataclass
class RetrievedChunk:
    text: str
    source_url: str
    source_title: str
    source_type: str
    source_tag: str
    score: float 


class Retriever:
    def __init__(self, embedder: Embedder,
                 db_path: str = DEFAULT_DB_PATH,
                 collection_name: str = DEFAULT_COLLECTION):
        self.embedder = embedder
        if not Path(db_path).exists():
            raise FileNotFoundError(f"ChromaDB không tồn tại: {db_path}. Chạy build_chromadb.py trước.")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        q_emb = self.embedder.embed([query])
        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=k,
        )
        chunks = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunks.append(RetrievedChunk(
                text=doc,
                source_url=meta.get("source_url", ""),
                source_title=meta.get("source_title", ""),
                source_type=meta.get("source_type", ""),
                source_tag=meta.get("source_tag", "") or "",
                score=float(dist),
            ))
        return chunks
