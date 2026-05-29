"""Chạy 6 experiments (ma trận 3 variations × 2 LLMs) trên test set.

Output: system_outputs/exp_{1a,1b,2a,2b,3a,3b}.txt
Mỗi file = 1 dòng/câu trả lời, tương ứng với data/test/questions.txt.

Ma trận:
    1a = Llama-3 + baseline
    1b = Qwen2.5 + baseline
    2a = Llama-3 + RAG zero-shot
    2b = Qwen2.5 + RAG zero-shot
    3a = Llama-3 + RAG few-shot
    3b = Qwen2.5 + RAG few-shot

Usage:
    # Chạy 1 experiment
    python experiments/run_experiments.py --exp 2a

    # Chạy tất cả với 1 LLM (tiết kiệm load/unload)
    python experiments/run_experiments.py --llm llama --variations 1,2,3
    python experiments/run_experiments.py --llm qwen --variations 1,2,3

    # Chạy hết 6 (auto load/unload)
    python experiments/run_experiments.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.embedder import Embedder
from pipeline.llm import load_llm
from pipeline.rag_pipeline import (
    clean_answer,
    variation_1_baseline,
    variation_2_rag_zero,
    variation_3_rag_few,
)
from pipeline.retriever import Retriever


EXP_MATRIX = {
    "1a": {"llm": "llama", "variation": 1},
    "1b": {"llm": "qwen", "variation": 1},
    "2a": {"llm": "llama", "variation": 2},
    "2b": {"llm": "qwen", "variation": 2},
    "3a": {"llm": "llama", "variation": 3},
    "3b": {"llm": "qwen", "variation": 3},
}


def load_questions(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_few_shot_examples(train_dir: str, n: int = 5) -> list[tuple[str, str]]:
    """Load n cặp Q&A chất lượng cao từ train set làm few-shot examples.

    Lọc:
    - Bỏ yes/no answers ("Có", "Không")
    - Bỏ answer > 8 từ (vi phạm rule trong system prompt)
    - Bỏ answer empty
    """
    qs = load_questions(f"{train_dir}/questions.txt")
    answs = load_questions(f"{train_dir}/reference_answers.txt")
    good = []
    for q, a in zip(qs, answs):
        a_lower = a.strip().lower()
        if a_lower in ("có", "không", "yes", "no"):
            continue
        a_words = len(a.split())
        if a_words > 8 or a_words < 1:
            continue
        good.append((q, a))
    return good[:n]


def run_experiment(
    exp_id: str,
    llm,  
    questions: list[str],
    retriever: Retriever | None,
    few_shot_examples: list[tuple[str, str]] | None,
    output_path: Path,
    resume: bool = False,
    k: int = 5,
) -> None:
    """Run 1 experiment, save output."""
    config = EXP_MATRIX[exp_id]
    variation = config["variation"]

    # Resume support
    answers = []
    start_idx = 0
    if resume and output_path.exists():
        existing = output_path.read_text(encoding="utf-8").splitlines()
        answers = existing
        start_idx = len(existing)
        print(f"  → Resume from question {start_idx}")

    n = len(questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = output_path.with_suffix(".log.jsonl")
    log_mode = "a" if resume else "w"
    log_f = log_path.open(log_mode, encoding="utf-8")

    t0 = time.time()
    try:
        with output_path.open("a" if resume else "w", encoding="utf-8") as fout:
            for i in range(start_idx, n):
                q = questions[i]
                try:
                    if variation == 1:
                        raw = variation_1_baseline(q, llm)
                        chunks = []
                    elif variation == 2:
                        raw, chunks = variation_2_rag_zero(q, llm, retriever, k=k)
                    else:  
                        raw, chunks = variation_3_rag_few(q, llm, retriever, few_shot_examples, k=k)
                    answer = clean_answer(raw)
                except Exception as e:
                    import traceback
                    print(f"  ✗ Error on Q{i+1}: {type(e).__name__}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    answer = ""
                    chunks = []
                    raw = ""

                fout.write(answer + "\n")
                fout.flush()

                log_f.write(json.dumps({
                    "idx": i,
                    "question": q,
                    "raw_output": raw,
                    "answer": answer,
                    "retrieved_chunk_ids": [c.source_url for c in chunks],
                }, ensure_ascii=False) + "\n")
                log_f.flush()

                answers.append(answer)
                if (i + 1) % 10 == 0 or (i + 1) == n:
                    elapsed = time.time() - t0
                    rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
                    eta = (n - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{n}] {rate:.1f} q/s, ETA {eta:.0f}s | Q: {q[:60]}")
                    print(f"            A: {answer[:80]}")
    finally:
        log_f.close()

    print(f"  Done {exp_id} in {time.time()-t0:.0f}s → {output_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", help="Chạy 1 experiment: 1a/1b/2a/2b/3a/3b")
    p.add_argument("--llm", choices=["llama", "qwen"],
                   help="Chạy tất cả variations với 1 LLM (tiết kiệm load)")
    p.add_argument("--variations", default="1,2,3",
                   help="Khi dùng --llm, chọn variations 1/2/3 (comma-separated)")
    p.add_argument("--all", action="store_true", help="Chạy hết 6 experiments")
    p.add_argument("--questions", default="data/test/questions.txt")
    p.add_argument("--train-dir", default="data/train",
                   help="Cho few-shot examples (variation 3)")
    p.add_argument("--output-dir", default="system_outputs")
    p.add_argument("--k", type=int, default=5, help="Top-k chunks retrieve")
    p.add_argument("--few-shot-n", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    # Determine which experiments to run
    if args.exp:
        exps_to_run = [args.exp]
    elif args.llm:
        variations = [int(v) for v in args.variations.split(",")]
        suffix = "a" if args.llm == "llama" else "b"
        exps_to_run = [f"{v}{suffix}" for v in variations]
    elif args.all:
        exps_to_run = list(EXP_MATRIX.keys())
    else:
        p.error("Phải có --exp, --llm, hoặc --all")

    print(f"═══ Will run {len(exps_to_run)} experiments: {exps_to_run} ═══\n")

    questions = load_questions(args.questions)
    print(f"→ {len(questions)} test questions\n")

    few_shot_examples = load_few_shot_examples(args.train_dir, args.few_shot_n)
    print(f"→ {len(few_shot_examples)} few-shot examples loaded\n")

    # Load embedder + retriever (chỉ cần 1 lần, dùng cho V2/V3)
    needs_retrieval = any(EXP_MATRIX[e]["variation"] in (2, 3) for e in exps_to_run)
    embedder = retriever = None
    if needs_retrieval:
        print(f"→ Loading embedder + retriever...")
        embedder = Embedder()
        retriever = Retriever(embedder)
        print(f"  ✓ Retriever ready\n")

    # Group exps by LLM để tiết kiệm load/unload
    exps_by_llm: dict[str, list[str]] = {}
    for e in exps_to_run:
        exps_by_llm.setdefault(EXP_MATRIX[e]["llm"], []).append(e)

    out_dir = Path(args.output_dir)

    for llm_name, exps in exps_by_llm.items():
        print(f"\n═══ Loading LLM: {llm_name} ═══")
        llm = load_llm(llm_name)
        try:
            for exp_id in exps:
                print(f"\n─── Experiment {exp_id} ───")
                out_path = out_dir / f"exp_{exp_id}.txt"
                run_experiment(
                    exp_id=exp_id,
                    llm=llm,
                    questions=questions,
                    retriever=retriever,
                    few_shot_examples=few_shot_examples,
                    output_path=out_path,
                    resume=args.resume,
                    k=args.k,
                )
        finally:
            print(f"\n→ Unloading {llm_name}")
            llm.unload()

    print(f"\n═══ ALL DONE ═══")
    print(f"Outputs trong {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
