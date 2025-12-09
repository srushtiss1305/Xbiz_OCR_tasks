# Indian Multi-OCR API

Flask API that runs 3 OCR engines in parallel:
- Tesseract (best for Hindi/Marathi)
- EasyOCR (native Marathi support)
- PaddleOCR (highest accuracy on printed docs)

## Features
- Supports images + PDF
- Returns text from all 3 engines
- Saves results to .txt files
- Works offline (CPU only)

## Quick Start
```bash
pip install -r requirements.txt
python main.py
→ http://127.0.0.1:8081/ocr