"""RAG pipeline — 3 variations.

- V1 (baseline): LLM trả lời trực tiếp, không retrieve
- V2 (RAG zero-shot): retrieve top-k → prompt + context
- V3 (RAG few-shot): V2 + 3-5 ví dụ Q&A từ train set
"""
from __future__ import annotations

from .llm import LLM
from .retriever import Retriever, RetrievedChunk


# ─────────────────────────────────────────────────────────
# System prompt — tối ưu cho factual short answer
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_BASE = (
    "Bạn là trợ lý trả lời câu hỏi về Đại học Quốc gia Hà Nội (VNU) và "
    "Trường Đại học Công nghệ (UET). "
    "Trả lời CHÍNH XÁC, NGẮN GỌN (tối đa 8 từ), KHÔNG giải thích thừa. "
    "Chỉ output câu trả lời, KHÔNG có 'Câu trả lời:' hay markdown."
)

SYSTEM_PROMPT_RAG = (
    "Bạn là trợ lý trả lời câu hỏi về Đại học Quốc gia Hà Nội (VNU) và "
    "Trường Đại học Công nghệ (UET). "
    "Sử dụng CONTEXT được cung cấp để trả lời. "
    "Trả lời CHÍNH XÁC, NGẮN GỌN (tối đa 8 từ), KHÔNG giải thích thừa. "
    "Nếu context không có thông tin, dùng kiến thức của bạn để đoán đáp án ngắn. "
    "Chỉ output câu trả lời, KHÔNG có 'Câu trả lời:' hay markdown."
)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Tài liệu {i}] ({c.source_title})\n{c.text}")
    return "\n\n".join(parts)


def _format_few_shot(examples: list[tuple[str, str]]) -> str:
    parts = []
    for q, a in examples:
        parts.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# V1 — Baseline (no retrieval)
# ─────────────────────────────────────────────────────────
def variation_1_baseline(question: str, llm: LLM, max_new_tokens: int = 64) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": f"Q: {question}\nA:"},
    ]
    return llm.generate(messages, max_new_tokens=max_new_tokens)


# ─────────────────────────────────────────────────────────
# V2 — RAG zero-shot
# ─────────────────────────────────────────────────────────
def variation_2_rag_zero(question: str, llm: LLM, retriever: Retriever,
                          k: int = 5, max_new_tokens: int = 64) -> tuple[str, list[RetrievedChunk]]:
    chunks = retriever.retrieve(question, k=k)
    context = _format_context(chunks)
    user_msg = f"CONTEXT:\n{context}\n\nQ: {question}\nA:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": user_msg},
    ]
    answer = llm.generate(messages, max_new_tokens=max_new_tokens)
    return answer, chunks


# ─────────────────────────────────────────────────────────
# V3 — RAG few-shot
# ─────────────────────────────────────────────────────────
def variation_3_rag_few(question: str, llm: LLM, retriever: Retriever,
                         examples: list[tuple[str, str]],
                         k: int = 5, max_new_tokens: int = 64) -> tuple[str, list[RetrievedChunk]]:
    chunks = retriever.retrieve(question, k=k)
    context = _format_context(chunks)
    few_shot = _format_few_shot(examples)
    user_msg = (
        f"CONTEXT:\n{context}\n\n"
        f"---\n\nVí dụ format câu hỏi-trả lời:\n\n{few_shot}\n\n"
        f"---\n\nQ: {question}\nA:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RAG},
        {"role": "user", "content": user_msg},
    ]
    answer = llm.generate(messages, max_new_tokens=max_new_tokens)
    return answer, chunks

def clean_answer(raw: str) -> str:
    text = raw.strip()
    text = text.split("\n")[0].strip()
    for prefix in ["A:", "a:", "Answer:", "Câu trả lời:", "Đáp án:", "Trả lời:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    text = text.strip("'\"`")
    return text
