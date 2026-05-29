from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HELP = """
Phím:
  [k] keep    — giữ nguyên
  [e] edit    — sửa câu hỏi/trả lời
  [d] drop    — bỏ
  [t] type    — đổi loại câu hỏi (1/2/3/4)
  [s] skip    — pause, review sau (lưu progress)
  [q] quit    — thoát (lưu progress)
  [?] help    — hiện help
"""


def print_pair(idx: int, total: int, pair: dict):
    print("\n" + "═" * 80)
    print(f"  [{idx+1}/{total}]  type={pair.get('type','?')}  source={pair.get('_source_tag', '-')}")
    print(f"  URL: {pair.get('_source_url', '')[:90]}")
    print("─" * 80)
    print(f"  Q: {pair.get('question', '')}")
    print(f"  A: {pair.get('answer', '')}")
    print(f"  Evidence: {pair.get('evidence', '')[:200]}")
    print("─" * 80)


def edit_pair(pair: dict) -> dict:
    print("\n→ Edit mode. Enter để giữ giá trị cũ.")
    q = input(f"  Q [{pair.get('question','')}]: ").strip()
    if q:
        pair["question"] = q
    a = input(f"  A [{pair.get('answer','')}]: ").strip()
    if a:
        pair["answer"] = a
    return pair


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/qa_draft.jsonl")
    p.add_argument("--output", default="data/qa_reviewed.jsonl")
    p.add_argument("--progress", default="data/.qa_review_progress.json")
    p.add_argument("--filter-type", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} không tồn tại. Chạy draft_qa.py trước.", file=sys.stderr)
        return 1

    pairs = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines()]
    if args.filter_type:
        pairs = [p for p in pairs if p.get("type") == args.filter_type]
    print(f"→ Loaded {len(pairs)} pairs to review")

    progress_path = Path(args.progress)
    start_idx = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and progress_path.exists():
        prog = json.loads(progress_path.read_text())
        start_idx = prog.get("next_idx", 0)
        print(f"→ Resume from index {start_idx}")

    kept = []
    if args.resume and out_path.exists():
        kept = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
        print(f"→ Existing reviewed: {len(kept)}")

    print(HELP)

    i = start_idx
    try:
        while i < len(pairs):
            pair = pairs[i]
            print_pair(i, len(pairs), pair)
            cmd = input("Action [k/e/d/t/s/q/?] (default k): ").strip().lower() or "k"

            if cmd == "?":
                print(HELP)
                continue
            elif cmd == "k":
                kept.append(pair)
                i += 1
            elif cmd == "e":
                pair = edit_pair(pair)
                kept.append(pair)
                i += 1
            elif cmd == "d":
                i += 1
            elif cmd == "t":
                new_type = input("  New type (1/2/3/4): ").strip()
                if new_type in {"1", "2", "3", "4"}:
                    pair["type"] = int(new_type)
                    kept.append(pair)
                    i += 1
                else:
                    print("  Invalid type, skipped")
            elif cmd == "s":
                print(f"→ Paused at index {i}. Resume với --resume")
                break
            elif cmd == "q":
                print(f"→ Quitting at index {i}. Resume với --resume")
                break
            else:
                print(f"  Unknown command: {cmd}")
    except (KeyboardInterrupt, EOFError):
        print(f"\n→ Interrupted at index {i}. Resume với --resume")

    with out_path.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    progress_path.write_text(json.dumps({"next_idx": i}, ensure_ascii=False))

    by_type = {}
    for k in kept:
        t = k.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\n═══ STATS ═══")
    print(f"  Reviewed: {i}/{len(pairs)}")
    print(f"  Kept: {len(kept)}")
    print(f"  By type: {by_type}")
    print(f"  Output: {out_path}")
    print(f"\nBước tiếp: python data_gen/export_qa.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
