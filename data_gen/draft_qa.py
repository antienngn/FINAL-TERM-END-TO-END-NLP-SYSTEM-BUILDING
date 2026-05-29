"""Draft Q&A pairs từ chunks bằng Claude API.

Mỗi chunk → Claude sinh 3-7 cặp Q&A có metadata (type 1-4, evidence, source_url).
Output: data/qa_draft.jsonl — sau đó human review qua manual_review.py.

Yêu cầu:
- ANTHROPIC_API_KEY trong env hoặc --api-key flag
- pip install anthropic

Usage:
    # Sample 300 chunks ngẫu nhiên (đủ tạo ~150 Q&A sau review)
    python data_gen/draft_qa.py --sample 300

    # Resume nếu interrupted (skip chunks đã done)
    python data_gen/draft_qa.py --sample 300 --resume

    # Dùng model rẻ hơn (Sonnet thay vì Opus)
    python data_gen/draft_qa.py --model claude-sonnet-4-6 --sample 300

Cost ước tính (claude-sonnet-4-6):
    300 chunks × ~800 input + ~600 output tokens ≈ $0.7-1.5 USD
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: pip install anthropic", file=sys.stderr)
    sys.exit(1)


SYSTEM_PROMPT = """Bạn là chuyên gia tạo dataset Q&A factual cho hệ thống RAG về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET).

Khi đọc 1 đoạn text, hãy tạo 3-7 cặp câu hỏi-trả lời CHẤT LƯỢNG CAO, theo yêu cầu:

**Format output (JSON array, không có gì khác):**
```json
[
  {
    "question": "Câu hỏi rõ ràng bằng tiếng Việt",
    "answer": "Trả lời ngắn (≤8 từ, tối ưu Exact Match metric)",
    "type": 1,
    "evidence": "Câu trong text gốc support câu trả lời này"
  }
]
```

**4 loại câu hỏi (PHỦ ĐỦ — cố gắng tạo đa dạng):**

- **type 1** — LLM có thể tự trả lời (kiến thức chung): "VNU ở thành phố nào?" → Hà Nội
- **type 2** — Cần tài liệu mới tốt hơn (specific facts): "Học phí UET ngành CNTT năm 2024 là bao nhiêu?"
- **type 3** — Chỉ trả lời được qua RAG (domain-specific): "Khoa CNTT UET thành lập năm nào?" → 1995
- **type 4** — Nhạy cảm thời gian (current/dated): "Năm 2026 UET tuyển sinh bao nhiêu ngành?"

**Tiêu chí chất lượng (BẮT BUỘC):**
1. Câu trả lời PHẢI có căn cứ trong text — KHÔNG hallucinate, KHÔNG suy luận.
2. Câu trả lời ngắn gọn (≤8 từ). Nếu số có đơn vị, viết gọn: "1995", "144 Xuân Thủy", "Hà Nội".
3. Nếu có nhiều đáp án đúng, phân tách bằng dấu chấm phẩy `;`. VD: "UET; ĐH Công nghệ".
4. Câu hỏi rõ ràng, không cần context bên ngoài. Đề cập tên cụ thể (VNU/UET, năm, ngành) trong câu hỏi.
5. Đa dạng entity: con người, năm, địa điểm, số liệu, sự kiện, tên ngành.
6. KHÔNG dùng "năm nay", "hiện tại" — ghi rõ năm cụ thể.
7. KHÔNG tạo câu hỏi quá generic hoặc câu hỏi mà text không trả lời rõ.

**Nếu text không có content factual chất lượng (chỉ là menu/listing/ads), trả về `[]`.**

Chỉ output JSON array, không kèm giải thích gì khác."""


USER_TEMPLATE = """Đoạn text từ {source_type} ({source_url}):

---
{text}
---

Tạo 3-7 cặp Q&A theo yêu cầu. Output JSON array."""


def load_done_chunk_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            done.add(d.get("_source_chunk_id"))
        except Exception:
            continue
    return done


def call_claude(client: Anthropic, model: str, chunk: dict, max_retries: int = 3):
    user_msg = USER_TEMPLATE.format(
        source_type=chunk.get("source_type", "unknown"),
        source_url=chunk["source_url"],
        text=chunk["text"],
    )

    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            # Strip code fences nếu có
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:].strip()
                raw = raw.rstrip("`").strip()
            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError(f"Expected list, got {type(pairs)}")
            return pairs, resp.usage
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  ✗ FAIL after {max_retries}: {type(e).__name__}: {e}", file=sys.stderr)
                return None, None
            time.sleep(2 ** attempt)
    return None, None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--output", default="data/qa_draft.jsonl")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="Anthropic model. Use claude-opus-4-7 for highest quality.")
    p.add_argument("--sample", type=int, default=None,
                   help="Random sample N chunks (default: all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="Skip chunks đã có trong output")
    p.add_argument("--api-key", default=None, help="Override ANTHROPIC_API_KEY env")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Stop after processing N chunks (for testing)")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Cần ANTHROPIC_API_KEY env hoặc --api-key", file=sys.stderr)
        return 1

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} không tồn tại. Chạy chunk_docs.py trước.", file=sys.stderr)
        return 1

    all_chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines()]
    print(f"→ Loaded {len(all_chunks)} chunks total")

    # Sample
    if args.sample and args.sample < len(all_chunks):
        rng = random.Random(args.seed)
        # Stratify by source_type để diversity
        by_type: dict[str, list] = {}
        for c in all_chunks:
            by_type.setdefault(c["source_type"], []).append(c)
        n_per_type = args.sample // len(by_type) + 1
        sampled = []
        for typ, lst in by_type.items():
            rng.shuffle(lst)
            sampled.extend(lst[:n_per_type])
        rng.shuffle(sampled)
        chunks_to_process = sampled[:args.sample]
        print(f"→ Sampled {len(chunks_to_process)} (stratified by source_type, seed={args.seed})")
    else:
        chunks_to_process = all_chunks

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_chunk_ids(out_path) if args.resume else set()
    if done_ids:
        print(f"→ Resume: skip {len(done_ids)} chunks đã done")
        chunks_to_process = [c for c in chunks_to_process if c["chunk_id"] not in done_ids]

    if args.max_chunks:
        chunks_to_process = chunks_to_process[:args.max_chunks]

    print(f"→ Will process {len(chunks_to_process)} chunks with {args.model}")

    client = Anthropic(api_key=api_key)
    total_in = total_out = 0
    total_pairs = 0

    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode, encoding="utf-8") as fout:
        for i, chunk in enumerate(chunks_to_process, 1):
            print(f"[{i}/{len(chunks_to_process)}] {chunk['chunk_id']} ({chunk['char_count']} chars)")
            pairs, usage = call_claude(client, args.model, chunk)
            if pairs is None:
                continue
            if usage:
                total_in += usage.input_tokens
                total_out += usage.output_tokens
            for pair in pairs:
                pair["_source_chunk_id"] = chunk["chunk_id"]
                pair["_source_url"] = chunk["source_url"]
                pair["_source_title"] = chunk.get("source_title", "")
                pair["_source_type"] = chunk["source_type"]
                pair["_source_tag"] = chunk.get("source_tag")
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            fout.flush()
            total_pairs += len(pairs)
            print(f"  ✓ Generated {len(pairs)} pairs")

    print(f"\n═══ DONE ═══")
    print(f"  Chunks processed: {len(chunks_to_process)}")
    print(f"  Q&A pairs drafted: {total_pairs}")
    print(f"  Tokens: in={total_in:,}, out={total_out:,}")
    print(f"  Output: {out_path}")
    print(f"\nBước tiếp: python data_gen/manual_review.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
