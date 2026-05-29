"""Evaluate predictions theo SQuAD-style metrics (Vietnamese-aware).

Metrics:
- EM (Exact Match): chuẩn hóa rồi so chính xác
- F1: token overlap (precision + recall harmonic mean)
- Answer Recall: % token reference xuất hiện trong prediction

Handle multi-answer: reference phân tách `;` → score = max trên các candidates.

Usage:
    # Evaluate 1 experiment
    python experiments/evaluate.py --pred system_outputs/exp_2a.txt

    # Evaluate tất cả + xuất CSV bảng so sánh
    python experiments/evaluate.py --all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_answer(s: str) -> str:
    """Normalize: lowercase + bỏ punctuation + collapse whitespace + NFC.

    Không strip articles (VN khác EN: không có a/an/the).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    for ch in "–—…“”‘’«»":
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()

def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def f1_score(pred: str, ref: str) -> float:
    pred_toks = get_tokens(pred)
    ref_toks = get_tokens(ref)
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def answer_recall(pred: str, ref: str) -> float:
    pred_toks = set(get_tokens(pred))
    ref_toks = get_tokens(ref)
    if not ref_toks:
        return 0.0
    return sum(1 for t in ref_toks if t in pred_toks) / len(ref_toks)

def best_metric_multi(pred: str, references: list[str], metric_fn) -> float:
    if not references:
        return 0.0
    return max(metric_fn(pred, ref) for ref in references)


def split_references(ref_line: str) -> list[str]:
    return [r.strip() for r in ref_line.split(";") if r.strip()]


def evaluate(predictions: list[str], references: list[str]) -> dict:
    assert len(predictions) == len(references), \
        f"Length mismatch: pred={len(predictions)} ref={len(references)}"

    em_list, f1_list, rec_list = [], [], []
    for pred, ref_line in zip(predictions, references):
        refs = split_references(ref_line)
        em_list.append(best_metric_multi(pred, refs, exact_match))
        f1_list.append(best_metric_multi(pred, refs, f1_score))
        rec_list.append(best_metric_multi(pred, refs, answer_recall))

    n = len(predictions)
    return {
        "n": n,
        "em": sum(em_list) / n * 100,
        "f1": sum(f1_list) / n * 100,
        "recall": sum(rec_list) / n * 100,
        "_per_instance": {
            "em": em_list,
            "f1": f1_list,
            "recall": rec_list,
        },
    }


def evaluate_by_type(predictions: list[str], references: list[str],
                      types: list[int]) -> dict:
    by_type = {}
    for t in [1, 2, 3, 4]:
        idxs = [i for i, ty in enumerate(types) if ty == t]
        if not idxs:
            continue
        preds_t = [predictions[i] for i in idxs]
        refs_t = [references[i] for i in idxs]
        by_type[t] = evaluate(preds_t, refs_t)
    return by_type


def load_question_types(qa_reviewed_path: str = "data/qa_reviewed.jsonl") -> dict[str, int]:
    p = Path(qa_reviewed_path)
    if not p.exists():
        return {}
    types = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        types[d.get("question", "").strip()] = d.get("type", 0)
    return types

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", help="Path tới file prediction (1 dòng/câu)")
    p.add_argument("--ref", default="data/test/reference_answers.txt")
    p.add_argument("--questions", default="data/test/questions.txt")
    p.add_argument("--all", action="store_true",
                   help="Evaluate all system_outputs/exp_*.txt")
    p.add_argument("--outputs-dir", default="system_outputs")
    p.add_argument("--csv-out", default="experiments/results.csv")
    p.add_argument("--per-type", action="store_true",
                   help="Show per-type breakdown")
    args = p.parse_args()

    references = Path(args.ref).read_text(encoding="utf-8").splitlines()
    questions = Path(args.questions).read_text(encoding="utf-8").splitlines() if Path(args.questions).exists() else []

    type_map = load_question_types() if args.per_type else {}
    types = [type_map.get(q.strip(), 0) for q in questions]

    if args.pred:
        files = [args.pred]
    elif args.all:
        files = sorted(glob.glob(f"{args.outputs_dir}/exp_*.txt"))
    else:
        p.error("Phải có --pred hoặc --all")

    if not files:
        raise SystemExit(f"Không tìm thấy file nào")

    print(f"{'Exp':<10} {'N':>5} {'EM':>8} {'F1':>8} {'Recall':>8}")
    print("─" * 50)

    rows = []
    for f in files:
        preds = Path(f).read_text(encoding="utf-8").splitlines()
        # Pad if shorter
        if len(preds) < len(references):
            preds = preds + [""] * (len(references) - len(preds))
        elif len(preds) > len(references):
            preds = preds[:len(references)]

        result = evaluate(preds, references)
        exp_name = Path(f).stem.replace("exp_", "")
        print(f"{exp_name:<10} {result['n']:>5} {result['em']:>7.2f}% {result['f1']:>7.2f}% {result['recall']:>7.2f}%")
        rows.append({"exp": exp_name, **{k: v for k, v in result.items() if k != "_per_instance"}})

        if args.per_type and types:
            by_type = evaluate_by_type(preds, references, types)
            for t, r in by_type.items():
                print(f"  └ type {t}: N={r['n']:>3} EM={r['em']:5.1f}% F1={r['f1']:5.1f}% Recall={r['recall']:5.1f}%")

    # Save CSV
    if rows:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=["exp", "n", "em", "f1", "recall"])
            w.writeheader()
            w.writerows(rows)
        print(f"\n→ CSV saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
