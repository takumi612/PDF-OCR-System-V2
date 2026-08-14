@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay .venv\Scripts\python.exe. Hay tao virtual environment theo README.md.
  exit /b 1
)
set "PYTHONPATH=src"
".venv\Scripts\python.exe" -m uvicorn government_ocr_text_api.main:app --app-dir src --host 127.0.0.1 --port 8000
