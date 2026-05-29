from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source_url: str
    source_title: str
    source_type: str
    source_tag: str | None
    chunk_index: int
    text: str
    char_count: int


def chunk_text(
    text: str,
    target_chars: int = 700,
    overlap_chars: int = 100,
    min_chunk_chars: int = 200,
) -> list[str]:
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > target_chars * 1.5:
            sentences = []
            buf = ""
            for ch in para:
                buf += ch
                if ch in ".!?\n" and len(buf) > 100:
                    sentences.append(buf.strip())
                    buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            parts = sentences
        else:
            parts = [para]

        for part in parts:
            if len(current) + len(part) + 2 <= target_chars:
                current = (current + "\n" + part).strip() if current else part
            else:
                if len(current) >= min_chunk_chars:
                    chunks.append(current)
                    if overlap_chars > 0 and len(current) > overlap_chars:
                        tail = current[-overlap_chars:]
                        space_idx = tail.find(" ")
                        if space_idx > 0:
                            tail = tail[space_idx:].strip()
                        current = (tail + "\n" + part).strip() if tail else part
                    else:
                        current = part
                else:
                    current = (current + "\n" + part).strip() if current else part

    if current and len(current) >= min_chunk_chars:
        chunks.append(current)
    elif current and chunks:
        chunks[-1] = chunks[-1] + "\n" + current

    return chunks


def process_file(
    jsonl_path: Path,
    target_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        text = d.get("text", "")
        if not text or d.get("content_length", 0) < min_chunk_chars:
            continue
        parts = chunk_text(text, target_chars, overlap_chars, min_chunk_chars)
        doc_id = d["id"]
        for i, ptext in enumerate(parts):
            out.append(Chunk(
                chunk_id=f"{doc_id}_{i:03d}",
                source_url=d["url"],
                source_title=d.get("title", ""),
                source_type=d.get("source_type", "other"),
                source_tag=d.get("tag"),
                chunk_index=i,
                text=ptext,
                char_count=len(ptext),
            ))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-glob", default="data/raw/*.jsonl")
    p.add_argument("--output", default="data/chunks.jsonl")
    p.add_argument("--target-chars", type=int, default=700)
    p.add_argument("--overlap-chars", type=int, default=100)
    p.add_argument("--min-chunk-chars", type=int, default=200)
    args = p.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"Không tìm thấy file nào match: {args.input_glob}")

    print(f"→ Chunking {len(files)} files (target ~{args.target_chars} chars, overlap {args.overlap_chars})")
    all_chunks: list[Chunk] = []
    for f in files:
        chunks = process_file(Path(f), args.target_chars, args.overlap_chars, args.min_chunk_chars)
        name = Path(f).name
        print(f"  [{name}] {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"\n→ Wrote {len(all_chunks)} chunks → {out_path}")
    if all_chunks:
        chars = [c.char_count for c in all_chunks]
        print(f"  chars/chunk: min={min(chars)}, max={max(chars)}, "
              f"avg={sum(chars)//len(chars)}, median={sorted(chars)[len(chars)//2]}")
        print(f"  total chars: {sum(chars):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
