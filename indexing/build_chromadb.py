from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERROR: pip install chromadb sentence-transformers — {e}", file=sys.stderr)
    sys.exit(1)


EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
COLLECTION_NAME = "vnu_uet_rag"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--db-path", default="indexing/chroma_db")
    p.add_argument("--collection", default=COLLECTION_NAME)
    p.add_argument("--model", default=EMBEDDING_MODEL)
    p.add_argument("--device", default="cuda",
                   help="cuda hoặc cpu (auto-detect nếu cuda fail)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="Max tokens/chunk (Vietnamese_Embedding hỗ trợ 2048)")
    p.add_argument("--reset", action="store_true",
                   help="Xóa collection cũ trước khi build")
    args = p.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} không tồn tại. Chạy chunk_docs.py trước.", file=sys.stderr)
        return 1

    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    device = args.device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("  CUDA not available, fallback to CPU")
                device = "cpu"
        except Exception:
            device = "cpu"
    print(f"Loading embedding model: {args.model} (device={device})")
    t0 = time.time()
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    dim = model.get_sentence_embedding_dimension()
    print(f"  Loaded in {time.time()-t0:.1f}s (dim={dim}, max_seq={args.max_seq_length})")

    db_path = Path(args.db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"  Deleted old collection '{args.collection}'")
        except Exception:
            pass

    try:
        collection = client.get_collection(args.collection)
        print(f"  Using existing collection '{args.collection}' (count={collection.count()})")
    except Exception:
        collection = client.create_collection(
            name=args.collection,
            metadata={"hnsw:space": "ip"},
        )
        print(f"  Created collection '{args.collection}' with hnsw:space=ip")

    print(f"→ Embedding & indexing {len(chunks)} chunks (batch={args.batch_size})")
    t0 = time.time()
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i:i + args.batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  
        )
        ids = [c["chunk_id"] for c in batch]
        metas = [
            {
                "source_url": c["source_url"],
                "source_title": c.get("source_title", "") or "",
                "source_type": c.get("source_type", "") or "",
                "source_tag": c.get("source_tag") or "",
                "chunk_index": int(c.get("chunk_index", 0)),
                "char_count": int(c.get("char_count", len(c["text"]))),
            }
            for c in batch
        ]
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metas,
        )
        done = min(i + args.batch_size, len(chunks))
        rate = done / (time.time() - t0)
        eta = (len(chunks) - done) / rate
        print(f"  [{done}/{len(chunks)}] {rate:.1f} chunks/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n→ Indexing done in {elapsed:.1f}s ({len(chunks)/elapsed:.1f} chunks/s)")
    print(f"  Collection count: {collection.count()}")
    print(f"  DB path: {db_path}")

    print(f"\n═══ VERIFY ═══")
    test_queries = [
        "Đại học Quốc gia Hà Nội thành lập năm nào?",
        "Khoa Công nghệ Thông tin UET",
        "Học phí UET năm 2025-2026",
    ]
    for q in test_queries:
        q_emb = model.encode([q], normalize_embeddings=False)
        res = collection.query(query_embeddings=q_emb.tolist(), n_results=3)
        print(f"\nQ: {q!r}")
        for rank, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            preview = doc.replace("\n", " ")[:120]
            print(f"  [{rank}] score={dist:.3f} | {meta.get('source_type')}/{meta.get('source_tag')}")
            print(f"      {preview}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
