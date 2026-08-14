FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0     && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN python -m pip install --no-cache-dir --upgrade pip     && python -m pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "government_ocr_text_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
