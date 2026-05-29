"""Paired bootstrap significance test giữa các cặp experiments.

Test giả thuyết: hiệu suất 2 hệ thống khác nhau (significant) hay không (chỉ noise).
Phương pháp: paired bootstrap với resampling.

Usage:
    # Compare 2 experiments
    python experiments/significance_test.py --a exp_1a --b exp_2a

    # All pairs (full matrix)
    python experiments/significance_test.py --all-pairs
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.evaluate import evaluate, exact_match, f1_score, best_metric_multi, split_references


def paired_bootstrap(
    preds_a: list[str],
    preds_b: list[str],
    references: list[str],
    metric_fn,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(references)

    # Per-instance scores
    refs_split = [split_references(r) for r in references]
    scores_a = [best_metric_multi(pa, refs, metric_fn) for pa, refs in zip(preds_a, refs_split)]
    scores_b = [best_metric_multi(pb, refs, metric_fn) for pb, refs in zip(preds_b, refs_split)]

    observed_diff = (sum(scores_b) - sum(scores_a)) / n

    # Bootstrap resamples
    diffs = []
    indices = list(range(n))
    for _ in range(n_iter):
        sample_idx = [rng.choice(indices) for _ in range(n)]
        sa = sum(scores_a[i] for i in sample_idx) / n
        sb = sum(scores_b[i] for i in sample_idx) / n
        diffs.append(sb - sa)

    diffs.sort()
    p_value = sum(1 for d in diffs if d <= 0) / n_iter
    ci_lower = diffs[int(0.025 * n_iter)]
    ci_upper = diffs[int(0.975 * n_iter)]

    return observed_diff, p_value, (ci_lower, ci_upper)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", help="Experiment A (vd: exp_1a)")
    p.add_argument("--b", help="Experiment B (vd: exp_2a)")
    p.add_argument("--all-pairs", action="store_true",
                   help="Test all pairs trong 6 experiments")
    p.add_argument("--outputs-dir", default="system_outputs")
    p.add_argument("--ref", default="data/test/reference_answers.txt")
    p.add_argument("--n-iter", type=int, default=1000)
    p.add_argument("--metric", default="f1", choices=["em", "f1"])
    args = p.parse_args()

    references = Path(args.ref).read_text(encoding="utf-8").splitlines()
    metric_fn = exact_match if args.metric == "em" else f1_score

    def load_preds(name: str) -> list[str]:
        name = name if name.startswith("exp_") else f"exp_{name}"
        path = Path(args.outputs_dir) / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8").splitlines()

    if args.all_pairs:
        exps = ["1a", "1b", "2a", "2b", "3a", "3b"]
        pairs = list(itertools.combinations(exps, 2))
    elif args.a and args.b:
        pairs = [(args.a, args.b)]
    else:
        p.error("Phải có --a + --b hoặc --all-pairs")

    print(f"Paired Bootstrap Test ({args.metric.upper()}, n_iter={args.n_iter})")
    print(f"{'A vs B':<15} {'B-A':>8} {'p-value':>10} {'95% CI':>20} {'Significant':>12}")
    print("─" * 75)

    for a_name, b_name in pairs:
        try:
            preds_a = load_preds(a_name)
            preds_b = load_preds(b_name)
        except FileNotFoundError as e:
            print(f"  ⊘ Skip {a_name} vs {b_name}: file không tồn tại")
            continue

        n = min(len(preds_a), len(preds_b), len(references))
        preds_a = preds_a[:n] + [""] * (n - len(preds_a)) if len(preds_a) < n else preds_a[:n]
        preds_b = preds_b[:n] + [""] * (n - len(preds_b)) if len(preds_b) < n else preds_b[:n]

        diff, p_val, (ci_lo, ci_hi) = paired_bootstrap(
            preds_a, preds_b, references[:n], metric_fn, args.n_iter,
        )
        sig = "✓ p<0.05" if p_val < 0.05 else ("~ p<0.10" if p_val < 0.10 else "✗")
        print(f"{a_name} vs {b_name:<8} {diff*100:>+7.2f}% {p_val:>10.4f} "
              f"[{ci_lo*100:+5.1f}%, {ci_hi*100:+5.1f}%] {sig:>12}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
