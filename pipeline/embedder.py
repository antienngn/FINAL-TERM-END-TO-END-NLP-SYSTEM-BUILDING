from __future__ import annotations

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "AITeamVN/Vietnamese_Embedding"
DEFAULT_MAX_SEQ = 2048


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: str = "cuda", max_seq_length: int = DEFAULT_MAX_SEQ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length

    def embed(self, texts: list[str]):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,  
            show_progress_bar=False,
        )
