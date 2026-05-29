# VNU/UET RAG System — Assignment 2

End-to-end Retrieval Augmented Generation system for factual QA về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET).

## 📊 Best results

| Variation | Best Exp | EM | F1 |
|---|---|---:|---:|
| V1 Baseline | 1b (Qwen2.5-7B) | 3.57% | 26.99% |
| **V2 RAG zero-shot** ⭐ | **2b (Qwen2.5-7B)** | **32.14%** | **59.86%** |
| V3 RAG few-shot | 3b (Qwen2.5-7B) | 31.63% | 59.34% |

→ RAG augmentation tăng EM ~9x so với baseline (paired bootstrap p<0.001).

## 🏗️ Architecture

```
Raw HTML/PDF → chunk (~700 chars) → embed (Vietnamese_Embedding, 1024-dim)
                                         ↓
                              ChromaDB (hnsw:space=ip)
                                         ↓
Question → embed → retrieve top-5 → LLM (Llama-3-8B / Qwen2.5-7B) → Answer
```

## 📂 Repository structure

```
RAG/
├── scrape/                    # Bước 1: Crawl HTML + PDF
│   ├── seeds.yaml             #   295 URLs (VNU/UET/PDF/Wiki)
│   ├── scrape_web.py
│   ├── scrape_pdf.py
│   └── run_full_crawl.sh
├── data_gen/                  # Bước 2: Q&A dataset
│   ├── chunk_docs.py
│   ├── prepare_qa_batches.py  #   Tạo batches → paste vào Claude
│   ├── merge_qa_responses.py
│   ├── auto_review.py
│   ├── manual_review.py
│   ├── clean_answers.py
│   └── export_qa.py
├── indexing/
│   └── build_chromadb.py
├── pipeline/                  # Bước 4: RAG pipeline
│   ├── embedder.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py        #   3 variations
├── experiments/               # Bước 5: Run + Eval
│   ├── run_experiments.py
│   ├── evaluate.py
│   └── significance_test.py
├── data/
│   ├── test/                  #   196 Q&A submission
│   └── train/                 #   30 Q&A submission
├── system_outputs/
│   ├── exp_{1a,1b,2a,2b,3a,3b}.txt
│   └── system_output_{1,2,3}.txt   ← Files nộp PDF
├── requirements.txt
└── README.md
```

## 🚀 Reproducibility — Quick start

### Prerequisites

- Conda env (Python 3.10)
- GPU: NVIDIA với CUDA 12.1 (tested: Tesla V100S 32GB)
- HuggingFace cache cho models:
  - `AITeamVN/Vietnamese_Embedding`
  - `meta-llama/Meta-Llama-3-8B-Instruct`
  - `Qwen/Qwen2.5-7B-Instruct`

### Setup

```bash
# 1. Create conda env
conda create -n myrag python=3.10 -y
conda activate myrag

# 2. Install deps
pip install -r requirements.txt

# 3. Fix nếu GPU không detect (driver CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run full pipeline (~30 phút trên V100)

```bash
# Bước 1 — Scrape ~236 docs (~5 min)
bash scrape/run_full_crawl.sh

# Bước 2 — Q&A dataset (manual paste vào Claude.ai)
python data_gen/chunk_docs.py
python data_gen/prepare_qa_batches.py --sample 50 --batch-size 50
# 👤 Paste data/qa_batches/batch_*.md vào Claude.ai
# 👤 Save response → data/qa_responses/batch_*.json
python data_gen/merge_qa_responses.py
python data_gen/auto_review.py
python data_gen/clean_answers.py
python data_gen/export_qa.py

# Bước 3 — Build ChromaDB (~5 min GPU)
python indexing/build_chromadb.py

# Bước 4 + 5 — Run 6 experiments + eval (~15 min)
python experiments/run_experiments.py --all
python experiments/evaluate.py --all --per-type
python experiments/significance_test.py --all-pairs
```

## 📋 Key design choices

| Decision | Rationale |
|---|---|
| **Embedding: AITeamVN/Vietnamese_Embedding** | Specialized cho VN, 1024-dim, dot product, max_seq=2048 |
| **ChromaDB `hnsw:space=ip`** | Match embedding model's dot product (KHÔNG cosine) |
| **Chunk size 700 chars (~250 tokens)** | Paragraph-aware, đủ context, không vượt 2048 max |
| **LLM dtype `float16`** | V100 không support bfloat16 |
| **Temperature 0 (greedy)** | Deterministic for factual QA |
| **Top-k = 5** | Balance precision-recall trên test set |
| **Few-shot: 5 examples, filtered** | Bỏ yes/no + answer >8 từ → cải thiện Qwen +4.5% F1 |
| **Claude as Q&A drafting assistant** | Drafted Q&A then human-reviewed (auto + manual tools) |

## 📊 Dataset stats

- **Knowledge base**: 236 docs, 2.2M chars, 4155 chunks
  - VNU main (vnu.edu.vn): ~42 pages
  - UET (uet.vnu.edu.vn): ~118 pages (incl. AI institute, curriculum, scholarships, tuition)
  - PDF regulations: 11 PDFs trích được text
  - Wikipedia VNU: 3 pages
- **Q&A dataset**: 226 pairs (196 test + 30 train)
  - Type 1 (general): 34 test
  - Type 2 (need docs): 33 test
  - Type 3 (RAG-only): 84 test
  - Type 4 (time-sensitive): 45 test

## ⚙️ Tech stack

| Layer | Tool |
|---|---|
| Scraping | `requests`, `beautifulsoup4`, `lxml`, `pdfplumber` |
| Embedding | `sentence-transformers` + `AITeamVN/Vietnamese_Embedding` |
| Vector DB | `chromadb` v1.5 (HNSW, dot product) |
| LLM | `transformers` + `accelerate` + Llama-3-8B / Qwen2.5-7B (fp16) |
| Evaluation | Custom (SQuAD-style, VN-aware normalize) |

## 📝 Reports

- [assignment2_task_list.md](assignment2_task_list.md) — 9-step roadmap (Vietnamese)
- `report.pdf` — Final report (≤7 pages, ACL template)

## 📜 Annotation methodology

**Hybrid:** Claude (Anthropic) used as **annotation drafting assistant**. For each chunk from crawled content, Claude proposed candidate Q&A pairs with structured metadata (type 1-4, source URL, supporting evidence). Each draft pair was then **reviewed and validated** by team via custom auto + interactive tools (`auto_review.py`, `manual_review.py`). Claude was NOT used in the RAG pipeline being evaluated (compliant with model policy: HuggingFace-only for evaluated systems).

## 🤝 Contributions

See `contributions.md`.

## 📄 License

Educational use only. Cite VNU/UET sources as listed in `scrape/seeds.yaml`.
