## Food Hazard Detection Demo (DS310)

Demo web **FastAPI (backend)** + **React/Vite (frontend)** cho bài toán Food Hazard Detection (ensemble DeBERTa + RoBERTa).

Repo này được cấu hình để **chỉ commit data tối thiểu phục vụ demo** (xem `.gitignore`).

### 1) Yêu cầu môi trường

- **Python**: 3.10+ (khuyến nghị 3.10/3.11)
- **Node.js**: 18+
- **Git**: để push lên GitHub

### 2) Chạy Backend (FastAPI)

Mở terminal tại thư mục root của project:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Backend chạy tại: `http://localhost:8000`

Các endpoint chính:
- **GET** `/health`
- **GET** `/validation?offset=0&limit=100`
- **POST** `/infer` với body:

```json
{ "title": "...", "content": "...", "top_k": 3 }
```

#### Data phục vụ tab Validation
Backend sẽ load theo thứ tự ưu tiên:
1) `data/validation_predictions.jsonl` (nếu có)  
2) `data/test_predictions.csv` (join với `data/incidents_valid.csv` theo `doc_id`)  
3) Fallback: hiển thị nhãn từ `data/incidents_valid.csv`

Tuỳ chọn (nếu muốn tạo `validation_predictions.jsonl` để demo mượt hơn):

```bash
python -m backend.precompute_validation --input data\incidents_valid.csv --output data\validation_predictions.jsonl --top-k 3
```

#### Hugging Face token (tuỳ chọn)
Nếu model repo cần quyền truy cập, set biến môi trường `HF_TOKEN` trước khi chạy backend.

### 3) Chạy Frontend (React/Vite)

Mở terminal khác:

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại: `http://localhost:5173`

Mặc định frontend gọi API ở `http://localhost:8000`. Nếu cần đổi, tạo file `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

### 4) Lưu ý quan trọng khi demo

- Lần đầu gọi `/infer` backend sẽ **tải model từ Hugging Face** → có thể mất vài phút tuỳ mạng.
- Nên giữ backend chạy để tận dụng cache model.



