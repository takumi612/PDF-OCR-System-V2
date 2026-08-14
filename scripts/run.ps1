$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m uvicorn government_ocr_text_api.main:app --host 127.0.0.1 --port 8000
