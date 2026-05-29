# Phân công công việc

## Thành viên

Bài tập này do một mình em thực hiện toàn bộ:

- Nguyễn Tiến An — Mã số sinh viên: 20021080 — Email: 20021080@vnu.edu.vn

Vì nhóm chỉ có một thành viên nên toàn bộ các phần dưới đây đều do em đảm nhận,
từ thu thập dữ liệu, chú thích dữ liệu, phát triển mô hình, chạy thực nghiệm cho
đến viết báo cáo.

## 1. Chú thích dữ liệu

Bộ dữ liệu gồm 226 cặp câu hỏi và câu trả lời, trong đó 196 cặp dùng cho tập
kiểm thử và 30 cặp dùng cho tập huấn luyện.

Quy trình chú thích: em dùng Claude của Anthropic để sinh câu hỏi và câu trả lời
từ dữ liệu đã thu thập, sau đó tự rà soát thủ công qua hai vòng. Vòng thứ nhất
đối chiếu từng câu trả lời với đoạn văn bằng chứng để bảo đảm tính chính xác.
Vòng thứ hai thống nhất định dạng, rút ngắn câu trả lời dài và chuẩn hóa các câu
có nhiều đáp án.

Toàn bộ 226 cặp (chỉ số từ 1 đến 226) đều do em chú thích và rà soát.

## 2. Thu thập dữ liệu và phát triển mô hình

Toàn bộ mã nguồn do em viết, được tổ chức theo từng bước như sau:

- Bước 1, thu thập dữ liệu: `scrape/scrape_web.py`, `scrape/scrape_pdf.py`,
  cùng tệp cấu hình `scrape/seeds.yaml`.
- Bước 2, tạo bộ câu hỏi và câu trả lời: `data_gen/chunk_docs.py`,
  `data_gen/prepare_qa_batches.py`, `data_gen/merge_qa_responses.py`,
  `data_gen/auto_review.py`, `data_gen/manual_review.py`,
  `data_gen/clean_answers.py`, `data_gen/export_qa.py`.
- Bước 3, xây dựng cơ sở dữ liệu vector: `indexing/build_chromadb.py`.
- Bước 4, xây dựng pipeline RAG: `pipeline/embedder.py`,
  `pipeline/retriever.py`, `pipeline/llm.py`, `pipeline/rag_pipeline.py`.
- Bước 5, chạy thực nghiệm và đánh giá: `experiments/run_experiments.py`,
  `experiments/evaluate.py`, `experiments/significance_test.py`.

## 3. Báo cáo và tài liệu

Em viết toàn bộ báo cáo (report.pdf) và tệp README.md hướng dẫn cách chạy.

## 4. Hướng dẫn chạy các tệp

Các bước dưới đây giả định môi trường đã được cài đặt theo `requirements.txt`.

### Bước 1: Thu thập dữ liệu

Thu thập dữ liệu từ các trang web và tệp PDF theo cấu hình trong `seeds.yaml`,
kết quả lưu vào thư mục `data/raw/`.

```
python scrape/scrape_web.py --seeds scrape/seeds.yaml
python scrape/scrape_pdf.py --seeds scrape/seeds.yaml
```

### Bước 2: Tạo bộ câu hỏi và câu trả lời

Chia tài liệu thành các đoạn nhỏ, chuẩn bị dữ liệu để đưa vào Claude, gộp kết
quả sinh được, rà soát rồi xuất ra tập kiểm thử và tập huấn luyện.

```
python data_gen/chunk_docs.py
python data_gen/prepare_qa_batches.py --sample 50 --batch-size 50
python data_gen/merge_qa_responses.py
python data_gen/auto_review.py
python data_gen/clean_answers.py
python data_gen/export_qa.py
```

Sau bước này, dữ liệu cuối cùng nằm trong `data/test/` và `data/train/`.

### Bước 3: Xây dựng cơ sở dữ liệu vector

Mã hóa các đoạn văn bản thành vector và lưu vào ChromaDB.

```
python indexing/build_chromadb.py
```

### Bước 4 và 5: Chạy thực nghiệm và đánh giá

Chạy cả sáu cấu hình thực nghiệm, sau đó tính các độ đo và kiểm định ý nghĩa
thống kê.

```
python experiments/run_experiments.py --all
python experiments/evaluate.py --all --per-type
python experiments/significance_test.py --all-pairs
```

Kết quả dự đoán của từng cấu hình được lưu trong thư mục `system_outputs/`, còn
bảng điểm tổng hợp được lưu tại `experiments/results.csv`.
