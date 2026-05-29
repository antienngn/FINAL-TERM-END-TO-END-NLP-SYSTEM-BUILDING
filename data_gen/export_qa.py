from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def write_qa_files(
    pairs: list[dict],
    out_dir: Path,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    q_path = out_dir / "questions.txt"
    a_path = out_dir / "reference_answers.txt"
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        for pair in pairs:
            q = pair["question"].strip().replace("\n", " ")
            a = pair["answer"].strip().replace("\n", " ")
            fq.write(q + "\n")
            fa.write(a + "\n")

    n_q = len(q_path.read_text(encoding="utf-8").splitlines())
    n_a = len(a_path.read_text(encoding="utf-8").splitlines())
    assert n_q == n_a == len(pairs), f"{label}: line count mismatch q={n_q} a={n_a} pairs={len(pairs)}"
    print(f"  ✓ {label}: {len(pairs)} pairs → {out_dir}/")


def stratified_split(pairs: list[dict], train_size: int, seed: int = 42) -> tuple[list, list]:
    rng = random.Random(seed)
    by_type: dict[int, list] = {}
    for p in pairs:
        by_type.setdefault(p.get("type", 0), []).append(p)

    per_type_train = train_size // max(len(by_type), 1) + 1
    train, test = [], []
    for typ, lst in by_type.items():
        rng.shuffle(lst)
        train.extend(lst[:per_type_train])
        test.extend(lst[per_type_train:])

    while len(train) < train_size and test:
        train.append(test.pop(0))
    if len(train) > train_size:
        overflow = train[train_size:]
        train = train[:train_size]
        test = overflow + test

    rng.shuffle(train)
    rng.shuffle(test)
    return test, train


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/qa_reviewed.jsonl")
    p.add_argument("--test-dir", default="data/test")
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--train-size", type=int, default=30)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"ERROR: {input_path} không tồn tại. Chạy manual_review.py trước.")

    pairs = [json.loads(l) for l in input_path.read_text(encoding="utf-8").splitlines()]
    print(f"→ Loaded {len(pairs)} reviewed Q&A pairs")

    by_type = {}
    for p in pairs:
        t = p.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"  By type: {by_type}")

    test, train = stratified_split(pairs, args.train_size, args.seed)
    if args.test_size and len(test) > args.test_size:
        test = test[:args.test_size]

    print(f"\n→ Split: train={len(train)}, test={len(test)}")

    write_qa_files(test, Path(args.test_dir), "test")
    write_qa_files(train, Path(args.train_dir), "train")

    print(f"\n═══ FINAL DATASET ═══")
    for name, ds in [("test", test), ("train", train)]:
        bt = {}
        for p in ds:
            t = p.get("type", "?")
            bt[t] = bt.get(t, 0) + 1
        print(f"  {name}: {len(ds)} pairs, by type: {bt}")

    print(f"\nFiles ready:")
    print(f"  {args.test_dir}/questions.txt + reference_answers.txt")
    print(f"  {args.train_dir}/questions.txt + reference_answers.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
