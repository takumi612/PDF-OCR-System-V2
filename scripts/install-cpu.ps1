$ErrorActionPreference = "Stop"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
Write-Host "Cài đặt xong. Hãy đặt models/vietocr/vgg_seq2seq.pth rồi chạy scripts/run.ps1"
