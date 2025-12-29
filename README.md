## Food Hazard Detection Demo (DS310)

Demo web **FastAPI (backend)** + **React/Vite (frontend)** cho bài toán Food Hazard Detection (ensemble DeBERTa + RoBERTa).

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


