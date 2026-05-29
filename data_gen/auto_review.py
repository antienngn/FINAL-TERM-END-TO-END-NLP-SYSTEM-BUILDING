from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter


def normalize_multi_answer(answer: str) -> str:
    if ";" in answer:
        return answer
    if answer.count(",") >= 2 and len(answer.split()) <= 12:
        parts = [p.strip() for p in answer.split(",") if p.strip()]
        return "; ".join(parts)
    return answer


def is_question_weak(q: str) -> tuple[bool, str]:
    words = q.strip().split()
    if len(words) < 5:
        return True, "câu hỏi quá ngắn (<5 từ)"
    if not q.strip().endswith("?"):
        return True, "không kết thúc bằng dấu ?"
    if "?" in q[:-1] and q.count("?") > 1:
        return True, "nhiều dấu ?"
    q_lower = q.lower()
    has_entity = any(kw in q_lower for kw in [
        "vnu", "uet", "đhqghn", "đại học quốc gia", "trường đại học công nghệ",
        "khoa", "viện", "ngành", "giáo sư", "gs", "ts", "bộ", "phòng",
        "năm", "năm 20", "2024", "2025", "2026", "wikipedia", "ai", "trí tuệ"
    ])
    if not has_entity and len(words) < 10:
        return True, "không có entity cụ thể (VNU/UET/năm/ngành)"
    return False, ""


def is_answer_weak(a: str, q: str, max_words: int = 12) -> tuple[bool, str]:
    a_clean = a.strip()
    if not a_clean:
        return True, "rỗng"
    words = a_clean.split()
    if len(words) > max_words:
        return True, f"quá dài ({len(words)} từ)"
    refusal = ["không có thông tin", "không rõ", "n/a", "không biết",
               "tôi không", "claude không", "không thể trả lời"]
    if any(r in a_clean.lower() for r in refusal):
        return True, "refusal/hallucination marker"
    return False, ""


def question_key(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip().rstrip("?"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/qa_draft.jsonl")
    p.add_argument("--output", default="data/qa_reviewed.jsonl")
    p.add_argument("--dropped-log", default="data/qa_dropped.jsonl")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"ERROR: {input_path} không tồn tại")

    pairs = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines()]
    print(f"→ Loaded {len(pairs)} pairs from {input_path}\n")

    kept = []
    dropped = []
    flagged_for_manual = []

    seen_questions = set()
    reasons = Counter()
    fixed_count = 0
    max_words = 8 if args.strict else 12

    for pair in pairs:
        q = pair.get("question", "")
        a = pair.get("answer", "")
        t = pair.get("type")

        qk = question_key(q)
        if qk in seen_questions:
            pair["_dropped_reason"] = "duplicate question"
            dropped.append(pair)
            reasons["duplicate"] += 1
            continue
        seen_questions.add(qk)

        q_weak, q_reason = is_question_weak(q)
        if q_weak:
            pair["_dropped_reason"] = f"question: {q_reason}"
            dropped.append(pair)
            reasons[f"q_{q_reason[:30]}"] += 1
            continue

        a_weak, a_reason = is_answer_weak(a, q, max_words=max_words)
        if a_weak:
            pair["_dropped_reason"] = f"answer: {a_reason}"
            dropped.append(pair)
            reasons[f"a_{a_reason[:30]}"] += 1
            continue

        new_a = normalize_multi_answer(a)
        if new_a != a:
            pair["answer"] = new_a
            pair["_auto_fixed"] = True
            fixed_count += 1

        n_words = len(new_a.split())
        if 8 < n_words <= max_words:
            pair["_flag_manual"] = f"answer {n_words} words (>8)"
            flagged_for_manual.append(pair)

        if t not in {1, 2, 3, 4}:
            pair["type"] = 3
            reasons["fix_type"] += 1

        kept.append(pair)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    drop_path = Path(args.dropped_log)
    with drop_path.open("w", encoding="utf-8") as f:
        for p in dropped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"═══ AUTO-REVIEW RESULTS ═══")
    print(f"  Input: {len(pairs)}")
    print(f"  Kept:  {len(kept)} ({len(kept)/len(pairs)*100:.0f}%)")
    print(f"  Dropped: {len(dropped)} ({len(dropped)/len(pairs)*100:.0f}%)")
    print(f"  Auto-fixed (multi-answer): {fixed_count}")
    print(f"  Flagged for manual review (9-{max_words} từ): {len(flagged_for_manual)}")

    print(f"\nDrop reasons:")
    for r, n in reasons.most_common():
        print(f"  {n:>3}× {r}")

    types = Counter(p.get("type") for p in kept)
    print(f"\nKept by type:")
    for t in [1, 2, 3, 4]:
        n = types.get(t, 0)
        pct = n / len(kept) * 100 if kept else 0
        print(f"  Type {t}: {n:>3} ({pct:.0f}%)")

    print(f"\n  Output kept:    {out_path}")
    print(f"  Output dropped: {drop_path}")

    if flagged_for_manual:
        print(f"\n⚠ {len(flagged_for_manual)} cặp flagged — chạy manual_review.py để manual review")
    else:
        print(f"\nBước tiếp: python data_gen/export_qa.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
