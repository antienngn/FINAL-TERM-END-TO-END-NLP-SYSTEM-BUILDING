from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM_INSTRUCTIONS_TYPE14_FOCUS = """Bạn là chuyên gia tạo dataset Q&A cho hệ thống RAG về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET).

🎯 **MỤC TIÊU ĐẶC BIỆT:** Lần này tập trung tạo câu hỏi **type 1 (kiến thức chung)** và **type 4 (nhạy cảm thời gian)**. Các batch trước đã đủ type 2 và 3.

Với MỖI chunk, hãy tạo 3-5 cặp Q&A, trong đó:
- **Ít nhất 1-2 cặp type 1** (kiến thức chung phổ biến mà LLM tự trả lời được)
- **Ít nhất 1-2 cặp type 4** (câu hỏi liên quan đến năm cụ thể / thời điểm / mới nhất)
- Có thể thêm 0-1 cặp type 2 hoặc 3 nếu chunk phù hợp

## Yêu cầu output

Output **DUY NHẤT** một JSON array. Mỗi entry phải có `chunk_id` chính xác.

```json
[
  {
    "chunk_id": "abc123_001",
    "question": "Câu hỏi rõ ràng bằng tiếng Việt",
    "answer": "Trả lời ngắn (≤8 từ)",
    "type": 1,
    "evidence": "Câu trong text gốc support câu trả lời"
  }
]
```

## 4 loại — CHI TIẾT type 1 và type 4

- **type 1** — Kiến thức chung phổ biến (LLM modern luôn biết). VD:
  - "Hà Nội là thủ đô của nước nào?" → Việt Nam
  - "VNU là viết tắt của gì?" → Vietnam National University
  - "Trí tuệ nhân tạo viết tắt là gì?" → AI
  - "Wikipedia là gì?" → Bách khoa toàn thư trực tuyến
  - "Khoa học máy tính nghiên cứu về cái gì?" → Máy tính và thuật toán
  - "Tiến sĩ là học vị cao nhất hay thấp nhất?" → Cao nhất
  - "Samsung là tập đoàn của nước nào?" → Hàn Quốc

- **type 4** — Nhạy cảm thời gian (câu trả lời gắn với năm/thời điểm cụ thể). VD:
  - "Điểm chuẩn UET ngành KHMT năm 2025 là bao nhiêu?" → 27.45
  - "Học bổng Toshiba 2025-2026 tại UET có bao nhiêu suất?"
  - "Xếp hạng QS World University 2026 của VNU?"
  - "Học phí UET năm học 2025-2026 cho CTĐT chuẩn?"
  - "Năm 2026 UET tuyển sinh bao nhiêu chỉ tiêu?"
  - "Lễ ra mắt Viện AI UET diễn ra năm nào?" → 2022
  - "QS Châu Á 2026 VNU xếp hạng thứ mấy?" → 158
  - "Olympic AI VOAI 2025 lần thứ mấy được tổ chức?" → Lần thứ nhất

- **type 2** — LLM biết đại khái nhưng doc cho câu trả lời chính xác hơn
- **type 3** — Chỉ doc mới trả lời được (chi tiết riêng biệt)

## Tiêu chí chất lượng (BẮT BUỘC)

1. **Câu trả lời PHẢI có căn cứ trong text** — KHÔNG hallucinate.
2. **Câu trả lời ≤8 từ**.
3. **Multi-answer phân tách bằng `;`**.
4. **Câu hỏi rõ ràng**, đề cập tên cụ thể (VNU/UET, năm, ngành).
5. **Type 4 phải có yếu tố thời gian rõ ràng** (năm, đợt, mùa).
6. **Type 1 phải là câu LLM modern thực sự biết** — không phải câu cần document.
7. **KHÔNG dùng "năm nay", "hiện tại"** — ghi rõ năm.

## Bắt đầu

Dưới đây là {n_chunks} chunks. Hãy tạo Q&A TẬP TRUNG type 1 và 4 — output JSON array."""


SYSTEM_INSTRUCTIONS_TYPE12_FOCUS = """Bạn là chuyên gia tạo dataset Q&A cho hệ thống RAG về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET).

🎯 **MỤC TIÊU ĐẶC BIỆT:** Lần này bạn cần tập trung tạo câu hỏi **type 1 và type 2**. Các batch trước đã đủ type 3 và type 4 rồi.

Với MỖI chunk, hãy tạo 3-5 cặp Q&A, trong đó:
- **Ít nhất 1-2 cặp type 1** (kiến thức chung mà LLM tự trả lời được)
- **Ít nhất 1-2 cặp type 2** (LLM biết đại khái nhưng cần tài liệu cho câu trả lời chính xác)
- Có thể có thêm 0-2 cặp type 3 hoặc 4 nếu chunk gợi ý rõ

## Yêu cầu output (CỰC KỲ QUAN TRỌNG)

Output **DUY NHẤT** một JSON array. Mỗi entry phải có `chunk_id` chính xác như tôi đưa.

```json
[
  {
    "chunk_id": "abc123_001",
    "question": "Câu hỏi rõ ràng bằng tiếng Việt",
    "answer": "Trả lời ngắn (≤8 từ)",
    "type": 1,
    "evidence": "Câu trong text gốc support câu trả lời"
  }
]
```

## 4 loại câu hỏi — CHI TIẾT phân biệt type 1 vs 2

- **type 1** — Kiến thức chung phổ biến: LLM modern (GPT/Claude/Llama) chắc chắn biết câu trả lời mà không cần document. VD:
  - "VNU ở thành phố nào?" → Hà Nội
  - "VNU là viết tắt của gì?" → Vietnam National University
  - "Wikipedia tiếng Việt thuộc dự án nào?" → Wikimedia
  - "Ngành Trí tuệ nhân tạo là gì?" → Lĩnh vực máy tính mô phỏng trí thông minh
  - "Đại học Quốc gia Hà Nội thuộc tỉnh nào?" → Hà Nội

- **type 2** — LLM biết đại khái nhưng document cho câu trả lời chính xác hơn / cập nhật hơn / chi tiết hơn. VD:
  - "VNU có khoảng bao nhiêu trường thành viên?" (LLM biết ~10, doc nói chính xác 9)
  - "Trường ĐH Công nghệ thành lập khoảng năm nào?" (LLM biết khoảng 2000s, doc nói 2004)
  - "Khoa CNTT của UET có bao nhiêu khoa con?" (LLM ước, doc nói chính xác)
  - "GS Chử Đức Trình giữ chức vụ gì ở UET?" (LLM có thể biết mơ hồ, doc xác nhận)
  - "AITeamVN Embedding hỗ trợ độ dài max bao nhiêu tokens?" (LLM có thể biết Vietnamese embedding ~2048, doc xác nhận)

- **type 3** — Chỉ document mới trả lời được (chi tiết riêng biệt): VD "Khoa CNTT UET thành lập ngày bao nhiêu?" → 11/02/1995

- **type 4** — Nhạy cảm thời gian: VD "Điểm chuẩn UET 2025?", "Học bổng Toshiba 2025-2026 tại UET bao nhiêu suất?"

## Tiêu chí chất lượng (BẮT BUỘC tuân thủ)

1. **Câu trả lời PHẢI có căn cứ trong text** — KHÔNG hallucinate.
2. **Câu trả lời ≤8 từ**. Số có đơn vị viết gọn: "1995", "Hà Nội", "9 trường", "12 triệu/năm".
3. **Multi-answer phân tách bằng `;`**. VD: "UET; ĐH Công nghệ".
4. **Câu hỏi rõ ràng, không cần context bên ngoài** — đề cập tên cụ thể (VNU/UET, năm, ngành) trong câu hỏi.
5. **Đa dạng entity**: con người, năm, địa điểm, số liệu, sự kiện.
6. **KHÔNG dùng "năm nay", "hiện tại"** — ghi rõ năm cụ thể.
7. **Nếu chunk không phù hợp tạo type 1 hoặc 2**, có thể skip chunk đó — KHÔNG ép tạo type 1/2 mà câu hỏi quá vô lý.

## Format input chunks

Mỗi chunk được đánh dấu `[CHUNK <id>]` rồi đến metadata + text. Đọc kỹ chunk_id và include CHÍNH XÁC vào output.

## Bắt đầu

Dưới đây là {n_chunks} chunks. Hãy tạo Q&A TẬP TRUNG type 1 và 2 — output JSON array."""


SYSTEM_INSTRUCTIONS = """Bạn là chuyên gia tạo dataset Q&A factual cho hệ thống RAG về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học Công nghệ (UET).

Tôi sẽ cung cấp cho bạn nhiều đoạn text (chunks). Với MỖI chunk, hãy tạo 3-7 cặp câu hỏi-trả lời CHẤT LƯỢNG CAO.

## Yêu cầu output (CỰC KỲ QUAN TRỌNG)

Output **DUY NHẤT** một JSON array. Mỗi entry phải có `chunk_id` chính xác như tôi đưa.

```json
[
  {
    "chunk_id": "abc123_001",
    "question": "Câu hỏi rõ ràng bằng tiếng Việt",
    "answer": "Trả lời ngắn (≤8 từ)",
    "type": 1,
    "evidence": "Câu trong text gốc support câu trả lời"
  },
  ...
]
```

## 4 loại câu hỏi (PHỦ ĐỦ)

- **type 1** — LLM có thể tự trả lời (kiến thức chung): VD "VNU ở thành phố nào?" → Hà Nội
- **type 2** — Cần tài liệu mới tốt hơn: VD "Học phí UET ngành CNTT năm 2024?"
- **type 3** — Chỉ trả lời được qua RAG (domain-specific): VD "Khoa CNTT UET thành lập năm nào?" → 1995
- **type 4** — Nhạy cảm thời gian: VD "Điểm chuẩn UET năm 2025?"

## Tiêu chí chất lượng (BẮT BUỘC tuân thủ)

1. **Câu trả lời PHẢI có căn cứ trong text** — KHÔNG hallucinate, KHÔNG suy luận.
2. **Câu trả lời ≤8 từ**. Số có đơn vị viết gọn: "1995", "144 Xuân Thủy", "Hà Nội", "12 triệu/năm".
3. **Multi-answer phân tách bằng `;`**. VD: "UET; ĐH Công nghệ".
4. **Câu hỏi rõ ràng, không cần context bên ngoài** — đề cập tên cụ thể (VNU/UET, năm, ngành) trong câu hỏi.
5. **Đa dạng entity**: con người, năm, địa điểm, số liệu, sự kiện, tên ngành.
6. **KHÔNG dùng "năm nay", "hiện tại"** — ghi rõ năm cụ thể.
7. **KHÔNG tạo câu hỏi quá generic** hoặc câu mà text không trả lời rõ.
8. **Nếu chunk không có content factual chất lượng** (chỉ là menu/listing/spam), bỏ qua chunk đó — KHÔNG tạo entry.

## Format input chunks

Mỗi chunk được đánh dấu `[CHUNK <id>]` rồi đến metadata + text. Đọc kỹ chunk_id và include CHÍNH XÁC vào output.

## Bắt đầu

Dưới đây là {n_chunks} chunks. Hãy tạo Q&A và output JSON array."""


def format_chunk(chunk: dict) -> str:
    return (
        f"[CHUNK {chunk['chunk_id']}]\n"
        f"Source: {chunk['source_url']}\n"
        f"Source type: {chunk.get('source_type','-')} | Tag: {chunk.get('source_tag','-')}\n"
        f"Text:\n"
        f"\"\"\"\n"
        f"{chunk['text']}\n"
        f"\"\"\""
    )


def stratified_sample(chunks: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for c in chunks:
        by_type.setdefault(c.get("source_type", "other"), []).append(c)

    n_per_type = n // len(by_type) + 1
    sampled = []
    for typ, lst in by_type.items():
        rng.shuffle(lst)
        sampled.extend(lst[:n_per_type])
    rng.shuffle(sampled)
    return sampled[:n]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--output-dir", default="data/qa_batches")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-from", default=None)
    p.add_argument("--focus-types", default=None, choices=["12", "14", None])
    p.add_argument("--start-batch-num", type=int, default=1)
    args = p.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"ERROR: {chunks_path} không tồn tại. Chạy chunk_docs.py trước.")

    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines()]
    print(f"→ Loaded {len(chunks)} chunks")

    if args.exclude_from:
        excl = set(json.loads(Path(args.exclude_from).read_text(encoding="utf-8")))
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in excl]
        print(f"→ Excluded {before - len(chunks)} chunks already sampled (from {args.exclude_from})")

    if args.sample < len(chunks):
        chunks = stratified_sample(chunks, args.sample, args.seed)
        print(f"→ Sampled {len(chunks)} (stratified by source_type)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_path = out_dir / "_sampled_chunks.json"
    existing = []
    if sampled_path.exists():
        try:
            existing = json.loads(sampled_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    new_ids = [c["chunk_id"] for c in chunks]
    combined = list(dict.fromkeys(existing + new_ids))
    sampled_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    print(f"→ Splitting into {n_batches} batches of {args.batch_size} chunks each")

    if args.focus_types == "12":
        prompt_template = SYSTEM_INSTRUCTIONS_TYPE12_FOCUS
        print(f"→ Using prompt: TYPE 1+2 FOCUS")
    elif args.focus_types == "14":
        prompt_template = SYSTEM_INSTRUCTIONS_TYPE14_FOCUS
        print(f"→ Using prompt: TYPE 1+4 FOCUS")
    else:
        prompt_template = SYSTEM_INSTRUCTIONS
        print(f"→ Using prompt: DEFAULT (all 4 types)")

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        batch = chunks[start:start + args.batch_size]

        prompt = prompt_template.replace("{n_chunks}", str(len(batch))) + "\n\n"
        prompt += "\n\n".join(format_chunk(c) for c in batch)

        batch_num = args.start_batch_num + batch_idx
        batch_file = out_dir / f"batch_{batch_num:03d}.md"
        batch_file.write_text(prompt, encoding="utf-8")

        total_chars = sum(c["char_count"] for c in batch)
        print(f"  ✓ batch_{batch_num:03d}.md — {len(batch)} chunks, ~{total_chars:,} chars text")

    print(f"\n═══ NEXT STEPS ═══")
    print(f"""
1. Mở Claude.ai (claude.ai/new) hoặc Claude khác — model **claude-sonnet-4-6** hoặc **claude-opus-4-7**

2. Với MỖI batch file ({out_dir}/batch_NNN.md):
   a. Mở file, copy TOÀN BỘ nội dung (Ctrl+A, Ctrl+C)
   b. Paste vào Claude → Claude sẽ trả về JSON array
   c. Copy JSON response → lưu thành: data/qa_responses/batch_NNN.json
      (Tạo folder data/qa_responses/ trước nếu chưa có)

3. Sau khi xong TẤT CẢ batches:
   python data_gen/merge_qa_responses.py

4. Tiếp theo: manual_review.py rồi export_qa.py

LƯU Ý:
- Nếu Claude trả response không phải JSON thuần (có markdown ```json fences), merge script sẽ auto-strip.
- Nếu Claude từ chối hoặc lỗi 1 batch, skip batch đó — không sao.
- Có thể chạy parallel: mở nhiều tab Claude và paste nhiều batches cùng lúc.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
