# Hệ thống RAG cho VNU và UET — Bài tập lớn 2

Hệ thống Sinh văn bản tăng cường truy xuất (Retrieval-Augmented Generation, RAG)
cho tác vụ trả lời câu hỏi về Đại học Quốc gia Hà Nội (VNU) và Trường Đại học
Công nghệ (UET).

Nguyễn Tiến An — Mã số sinh viên 20021080.

## Kết quả chính

Bảng dưới đây là kết quả tốt nhất của mỗi biến thể trên tập kiểm thử 196 câu hỏi.

| Biến thể | Cấu hình tốt nhất | Exact Match | F1 |
|---|---|---:|---:|
| Trả lời trực tiếp | Qwen2.5-7B | 3,57% | 26,99% |
| RAG không ví dụ | Qwen2.5-7B | 32,14% | 59,86% |
| RAG có ví dụ | Qwen2.5-7B | 31,63% | 59,34% |

Việc bổ sung ngữ cảnh truy xuất giúp điểm Exact Match tăng khoảng chín lần so với
mô hình trả lời trực tiếp, và mức cải thiện này có ý nghĩa thống kê rõ rệt qua
kiểm định tự lấy mẫu theo cặp với giá trị p nhỏ hơn 0,001.

## Kiến trúc hệ thống

```
Tài liệu HTML và PDF
    -> chia đoạn (khoảng 700 ký tự)
    -> mã hóa vector (AITeamVN/Vietnamese_Embedding, 1024 chiều)
    -> lưu vào ChromaDB (không gian tích vô hướng)

Câu hỏi
    -> mã hóa vector
    -> truy xuất 5 đoạn liên quan nhất từ ChromaDB
    -> đưa vào mô hình ngôn ngữ (Llama-3-8B hoặc Qwen2.5-7B)
    -> câu trả lời
```

## Cấu trúc thư mục

```
RAG/
├── scrape/              Bước 1: thu thập dữ liệu HTML và PDF
│   ├── seeds.yaml       danh sách URL cần thu thập
│   ├── scrape_web.py
│   ├── scrape_pdf.py
│   └── run_full_crawl.sh
├── data_gen/            Bước 2: tạo bộ câu hỏi và câu trả lời
│   ├── chunk_docs.py
│   ├── prepare_qa_batches.py
│   ├── merge_qa_responses.py
│   ├── auto_review.py
│   ├── manual_review.py
│   ├── clean_answers.py
│   └── export_qa.py
├── indexing/           Bước 3: xây dựng cơ sở dữ liệu vector
│   └── build_chromadb.py
├── pipeline/           Bước 4: pipeline RAG với ba biến thể
│   ├── embedder.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
├── experiments/        Bước 5: chạy thực nghiệm và đánh giá
│   ├── run_experiments.py
│   ├── evaluate.py
│   └── significance_test.py
├── data/
│   ├── test/           196 cặp câu hỏi và câu trả lời để nộp
│   └── train/          30 cặp dùng làm ví dụ minh họa
├── system_outputs/     kết quả dự đoán của các cấu hình
├── requirements.txt
└── README.md
```

## Cài đặt môi trường

Hệ thống chạy trên Python 3.10 và sử dụng GPU NVIDIA. Quá trình phát triển được
thực hiện trên card Tesla V100 với CUDA 12.1.

```
conda create -n myrag python=3.10 -y
conda activate myrag
pip install -r requirements.txt
```

Nếu GPU không được nhận do phiên bản trình điều khiển CUDA, hãy cài lại PyTorch
cho đúng phiên bản CUDA của máy, ví dụ với CUDA 12.1:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Hệ thống cần ba mô hình sau từ Hugging Face. Lần chạy đầu tiên các mô hình sẽ
được tải về tự động:

- AITeamVN/Vietnamese_Embedding
- meta-llama/Meta-Llama-3-8B-Instruct
- Qwen/Qwen2.5-7B-Instruct

## Hướng dẫn chạy

Toàn bộ quy trình mất khoảng ba mươi phút trên GPU V100.

### Bước 1: Thu thập dữ liệu

Thu thập dữ liệu từ các trang web và tệp PDF theo cấu hình trong seeds.yaml,
kết quả lưu vào thư mục data/raw.

```
bash scrape/run_full_crawl.sh
```

### Bước 2: Tạo bộ câu hỏi và câu trả lời

Trước hết chia tài liệu thành các đoạn nhỏ và chuẩn bị dữ liệu để đưa vào Claude.

```
python data_gen/chunk_docs.py
python data_gen/prepare_qa_batches.py --sample 50 --batch-size 50
```

Tiếp theo, mở từng tệp trong data/qa_batches dán vào Claude, rồi lưu phản hồi
vào data/qa_responses dưới dạng tệp JSON tương ứng. Sau khi có phản hồi, chạy
các lệnh sau để gộp, rà soát và xuất ra dữ liệu cuối cùng.

```
python data_gen/merge_qa_responses.py
python data_gen/auto_review.py
python data_gen/clean_answers.py
python data_gen/export_qa.py
```

Kết quả nằm trong data/test và data/train.

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

Kết quả dự đoán của từng cấu hình nằm trong thư mục system_outputs, còn bảng
điểm tổng hợp nằm tại experiments/results.csv.

## Các lựa chọn thiết kế quan trọng

| Lựa chọn | Lý do |
|---|---|
| Mô hình mã hóa AITeamVN/Vietnamese_Embedding | Tối ưu cho tiếng Việt, vector 1024 chiều, độ dài tối đa 2048 token |
| ChromaDB dùng không gian tích vô hướng | Khớp với cách đo độ tương đồng của mô hình mã hóa, không dùng cosine |
| Kích thước đoạn khoảng 700 ký tự | Giữ trọn đoạn văn, đủ ngữ cảnh và không vượt giới hạn token |
| Mô hình ngôn ngữ chạy ở float16 | Phù hợp bộ nhớ của GPU V100 |
| Giải mã tham lam | Bảo đảm kết quả tái lập qua nhiều lần chạy |
| Truy xuất 5 đoạn | Cân bằng giữa đủ thông tin và tránh làm loãng ngữ cảnh |
| Năm ví dụ minh họa đã lọc | Bỏ câu trả lời dạng có hoặc không và câu quá dài để tăng chất lượng |
| Dùng Claude để sinh dữ liệu | Chỉ dùng khi chú thích, không tham gia hệ thống được chấm điểm |

## Thống kê dữ liệu

Kho tri thức gồm 236 tài liệu, khoảng 2,2 triệu ký tự, chia thành 4155 đoạn:

- Cổng thông tin VNU: khoảng 42 trang
- Trường Đại học Công nghệ: khoảng 118 trang, gồm các khoa và viện, tuyển sinh,
  học bổng, học phí và Viện Trí tuệ Nhân tạo
- Văn bản quy chế dạng PDF: 11 tệp trích được văn bản
- Wikipedia về VNU: 3 trang

Bộ câu hỏi và câu trả lời gồm 226 cặp, trong đó 196 cặp cho tập kiểm thử và 30
cặp cho tập huấn luyện, phủ đủ bốn loại câu hỏi:

- Loại 1, mô hình tự trả lời được: 34 câu
- Loại 2, cần tài liệu để trả lời chính xác hơn: 33 câu
- Loại 3, chỉ trả lời được nhờ truy xuất: 84 câu
- Loại 4, nhạy cảm với yếu tố thời gian: 45 câu

## Công nghệ sử dụng

| Thành phần | Thư viện và mô hình |
|---|---|
| Thu thập dữ liệu | requests, beautifulsoup4, lxml, pdfplumber |
| Mã hóa vector | sentence-transformers với AITeamVN/Vietnamese_Embedding |
| Cơ sở dữ liệu vector | chromadb (thuật toán HNSW, tích vô hướng) |
| Mô hình ngôn ngữ | transformers, accelerate với Llama-3-8B và Qwen2.5-7B ở float16 |
| Đánh giá | Mã tự viết theo chuẩn SQuAD, có chuẩn hóa cho tiếng Việt |

## Phương pháp chú thích dữ liệu

Em dùng Claude của Anthropic để sinh câu hỏi và câu trả lời từ mỗi đoạn văn bản
đã thu thập, mỗi cặp đi kèm loại câu hỏi, đường dẫn nguồn và đoạn văn bằng chứng.
Sau đó em tự rà soát thủ công qua hai vòng, đối chiếu với bằng chứng và chuẩn hóa
định dạng. Claude chỉ được dùng trong giai đoạn chú thích dữ liệu và không tham
gia vào hệ thống RAG được đánh giá, do đó tuân thủ yêu cầu chỉ dùng mô hình mã
nguồn mở cho phần được chấm điểm.

## Phân công công việc

Bài tập do một mình em thực hiện. Chi tiết xem trong tệp contributions.md.
