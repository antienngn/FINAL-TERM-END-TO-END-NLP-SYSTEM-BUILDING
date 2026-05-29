from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ADDRESS_KEYWORDS = [
    "phòng ", "số ", "đường ", "quận ", "huyện ", "tỉnh ",
    "hà nội", "tp.", "tp ", "tỉnh ",
    "xuân thủy", "cầu giấy", "ba đình", "đống đa",
    "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8",
    "g1", "g2", "g3", "h1", "h2",
]


def is_address(answer: str) -> bool:
    a_lower = answer.lower()
    matches = sum(1 for kw in ADDRESS_KEYWORDS if kw in a_lower)
    if matches >= 2:
        return True
    if re.search(r"\d+\s*[a-zA-Z]?\s*(xuân|cầu|nguyễn|trần|lê|phạm)", a_lower):
        return True
    return False


def normalize_multi_answer(answer: str) -> tuple[str, bool]:
    if ";" in answer:
        return answer, False
    if answer.count(",") < 2:
        return answer, False
    if is_address(answer):
        return answer, False

    parts = [p.strip() for p in answer.split(",") if p.strip()]
    if not parts:
        return answer, False
    if any(len(p.split()) > 7 for p in parts):
        return answer, False
    sentence_words = [" thuộc ", " tại ", " của ", " được ", " đã ", " là ",
                      " trong ", " với ", " và ", " hoặc ", " có ", " sẽ ",
                      " bao gồm ", " gồm "]
    if any(sw in answer.lower() for sw in sentence_words):
        return answer, False
    new = "; ".join(parts)
    return new, True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/qa_reviewed.jsonl")
    p.add_argument("--output", default="data/qa_reviewed.jsonl")
    p.add_argument("--shorten-descriptive", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    pairs = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines()]
    print(f"→ Loaded {len(pairs)} pairs\n")

    converted_count = 0
    dropped_descriptive = 0
    kept = []

    for pair in pairs:
        a = pair.get("answer", "")
        new_a, was_changed = normalize_multi_answer(a)
        if was_changed:
            converted_count += 1
            pair["answer"] = new_a
            pair["_normalized_multi"] = True

        if args.shorten_descriptive:
            words = pair["answer"].split()
            if len(words) > 8 and ";" not in pair["answer"]:
                dropped_descriptive += 1
                continue

        kept.append(pair)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"═══ STATS ═══")
    print(f"  Multi-answer converted (',' → ';'): {converted_count}")
    if args.shorten_descriptive:
        print(f"  Dropped descriptive long (>8 từ): {dropped_descriptive}")
    print(f"  Final pairs: {len(kept)}")
    print(f"  Output: {out_path}")
    print(f"\nBước tiếp: python data_gen/export_qa.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
