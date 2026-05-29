from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path


def parse_response(content: str) -> list[dict] | None:
    content = content.strip()
    if content.startswith("```"):
        first = content.find("\n", 3)
        last = content.rfind("```")
        if first > 0 and last > first:
            content = content[first:last].strip()
        else:
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            return None
        return data
    except json.JSONDecodeError as e:
        m = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        print(f"  ✗ Parse failed: {e}", file=sys.stderr)
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--responses-dir", default="data/qa_responses")
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--output", default="data/qa_draft.jsonl")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"ERROR: {chunks_path} không tồn tại")
    chunks_by_id = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        chunks_by_id[c["chunk_id"]] = c
    print(f"→ Loaded {len(chunks_by_id)} chunks lookup")

    responses_dir = Path(args.responses_dir)
    if not responses_dir.exists():
        raise SystemExit(
            f"ERROR: {responses_dir} không tồn tại.\n"
            f"Tạo folder + lưu response Claude vào {responses_dir}/batch_001.json, batch_002.json, ..."
        )

    files = sorted(glob.glob(str(responses_dir / "batch_*.json"))) + \
            sorted(glob.glob(str(responses_dir / "batch_*.txt")))
    if not files:
        raise SystemExit(f"ERROR: Không tìm thấy batch_*.json/txt trong {responses_dir}")

    print(f"→ Found {len(files)} response files")

    all_pairs = []
    bad_chunks = []
    skipped_files = []
    for f in files:
        content = Path(f).read_text(encoding="utf-8")
        pairs = parse_response(content)
        name = Path(f).name
        if pairs is None:
            print(f"  ✗ {name}: parse failed")
            skipped_files.append(name)
            continue

        valid = 0
        for pair in pairs:
            required = {"chunk_id", "question", "answer", "type"}
            if not required.issubset(pair.keys()):
                if args.strict:
                    raise SystemExit(f"  ✗ {name}: missing fields in {pair}")
                continue

            chunk_id = pair["chunk_id"]
            if chunk_id not in chunks_by_id:
                bad_chunks.append(chunk_id)
                if args.strict:
                    raise SystemExit(f"  ✗ {name}: unknown chunk_id {chunk_id}")
                continue

            chunk = chunks_by_id[chunk_id]
            pair["_source_chunk_id"] = chunk_id
            pair["_source_url"] = chunk["source_url"]
            pair["_source_title"] = chunk.get("source_title", "")
            pair["_source_type"] = chunk["source_type"]
            pair["_source_tag"] = chunk.get("source_tag")
            all_pairs.append(pair)
            valid += 1
        print(f"{name}: {valid}/{len(pairs)} valid pairs")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    by_type = {}
    by_source = {}
    for p in all_pairs:
        t = p.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
        s = p.get("_source_type", "?")
        by_source[s] = by_source.get(s, 0) + 1

    print(f"\n═══ STATS ═══")
    print(f"  Total Q&A pairs: {len(all_pairs)}")
    print(f"  By type: {by_type}")
    print(f"  By source: {by_source}")
    if bad_chunks:
        print(f"  Skipped {len(bad_chunks)} pairs with unknown chunk_id")
    if skipped_files:
        print(f"  Skipped {len(skipped_files)} files: {skipped_files}")
    print(f"\n  Output: {out_path}")
    print(f"\nBước tiếp: python data_gen/manual_review.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
