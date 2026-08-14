from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("government_ocr_text_api.main:app", host="127.0.0.1", port=8000)
