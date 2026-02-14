## Demo
<p align="center">
  <img src="demo.gif" width="600">
</p>

## Cài dặt và sử dụng
### 1. Clone repo
```text
git clone https://github.com/HoangLeminh17/Chatbot_tuyen_sinh_FPTU_2026.git
cd Chatbot_tuyen_sinh_FPTU_2026
```
### 2. Tạo môi trường - cài đặt thư viện
```text
conda create -n chatbot python=3.10
conda activate chatbot
```

```text
pip install -r requirements.txt
```

### 3. Chạy
- Cần chạy xong ingest_database.py mới có folder chroma_db
```text
python ingest_database.py  
python chatbot.py   
```


## Cấu trúc repo
```text
Chatbot-with-RAG-and-LangChain/
│
├── data/ # Nguồn data thô chứa các file .txt và .pdf
├── chroma_db/ # Vector database
├── chatbot.py
├── ingest_database.py # Xây vector database
├── .env # API key
└── README.md
```

## Các bước thực hiện:
1. Thu thập dữ liệu - Tiền xử lý dữ liệu
- Dữ liệu tuyển sinh của trường Đại học FPT năm 2026 được thu thập trên trang web của [trường Đại học FPT](https://daihoc.fpt.edu.vn/tuyen-sinh/)
- Dữ liệu tuyển sinh của trường Đại học FPT năm 2025 được thu thập trên trang web của [tuyensinh247](https://diemthi.tuyensinh247.com/de-an-tuyen-sinh/dai-hoc-fpt-FPT.html#file-pdf-de-an)
- Tất cả các dữ liệu được thu thập vào thời điểm: 12/02/2026
- Dữ liệu được thu thập dưới dạng .txt và .pdf qua các trang web trên
- Các file .txt được format lại cho phù hợp bài toán RAG

2. Mô hình và triển khai
- Source code gốc: [ThomasJanssen-tech](https://github.com/ThomasJanssen-tech/Chatbot-with-RAG-and-LangChain)
- Project gốc được cấp phép dưới MIT License
- Code được phát triển, cải tiến để phù hợp cho bài toán tư vấn tuyển sinh của trường Đại học FPT
- Mô hình embedding sử dụng: `BAAI/bge-m3` đa ngôn ngữ, phù hợp cho tiếng Việt.
- Mô hình ngôn ngữ sử dụng: `deepseek/deepseek-r1-0528:free`, sử dụng qua API Key của [Open Router](https://openrouter.ai/)

## Lấy API
- Truy cập https://openrouter.ai/settings/keys để lấy API key
- Tạo file ".env" trong repo, thiết lập OPENROUTER_API_KEY = API Key của bạn
- Bạn có thể thử tìm kiếm các mô hình ngỗn ngữ khác phù hợp API của bạn: https://openrouter.ai/models
