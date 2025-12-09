# OCR API - Multi-Engine Text Extraction

A Flask-based OCR API that supports multiple OCR engines (Tesseract, EasyOCR, PaddleOCR) with flexible input options (base64 or file path).

## 🚀 Features

- **Multiple OCR Engines**: Choose between Tesseract, EasyOCR, or PaddleOCR
- **Flexible Input**: Accepts both base64 encoded images and file paths
- **Multi-format Support**: PNG, JPG, JPEG, TIFF, PDF, and more
- **Text File Export**: Automatically saves extracted text to `.txt` files
- **Base64 Response**: Returns base64 encoded image in response
- **Modular Architecture**: Separate modules for each OCR engine

## 📋 Prerequisites

- Python 3.12.1
- Tesseract OCR (system installation required)

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd task-2_ocr
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (System Level)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and add to PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

## 📁 Project Structure

```
task-2_ocr/
│
├── main.py              # Main Flask application
├── tesseract_ocr.py     # Tesseract OCR module
├── easyocr_ocr.py       # EasyOCR module
├── paddle_ocr.py        # PaddleOCR module
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
│
├── images/              # Temporary image storage
├── ocr_results/         # Extracted text files (.txt)
└── .venv/               # Virtual environment
```

## 🎯 Usage

### 1. Start the Flask server
```bash
python main.py
```

Server will start at: `http://127.0.0.1:8081`

### 2. API Endpoints

#### **POST /ocr** - Extract text from image

**Request Body (JSON):**

**Option 1: Using File Path**
```json
{
  "image": "C:/Users/YourName/Desktop/document.jpg",
  "ocr_id": 1,
  "documentType": "invoice"
}
```

**Option 2: Using Base64**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAY...",
  "ocr_id": 2,
  "documentType": "receipt"
}
```

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | string | Yes | File path OR base64 encoded image |
| `ocr_id` | integer | Yes | OCR engine: `1` = Tesseract, `2` = EasyOCR, `3` = PaddleOCR |
| `documentType` | string | No | Optional document type label |

**Response:**
```json
{
  "input_type": "file_path",
  "documentType": "invoice",
  "ocr_engine": "Tesseract",
  "ocr_id": 1,
  "base64_image": "iVBORw0KGgoAAAANS...",
  "extracted_text": "Invoice\nTotal Amount: $500\n...",
  "saved_file": "ocr_results/tesseract_abc123.txt",
  "msg": "OCR completed using Tesseract",
  "remark": "success"
}
```

#### **GET /health** - Health check

**Response:**
```json
{
  "status": "healthy",
  "available_engines": {
    "1": "Tesseract",
    "2": "EasyOCR",
    "3": "PaddleOCR"
  }
}
```

## 🧪 Testing with Postman

### Step 1: Create new POST request
- Method: `POST`
- URL: `http://127.0.0.1:8081/ocr`

### Step 2: Set Headers
- `Content-Type: application/json`

### Step 3: Set Body
- Select `raw` and `JSON`
- Paste your JSON payload

### Step 4: Send Request
Click **Send** and view the response!

## 🔧 OCR Engines

### 1. Tesseract (ocr_id: 1)
- **Best for:** Printed documents, clear text
- **Languages:** English, Hindi, Marathi
- **Speed:** Fast
- **Preprocessing:** Yes (CLAHE, thresholding, upscaling)

### 2. EasyOCR (ocr_id: 2)
- **Best for:** Natural scenes, handwriting
- **Languages:** English, Marathi
- **Speed:** Medium
- **Preprocessing:** No

### 3. PaddleOCR (ocr_id: 3)
- **Best for:** Multi-language documents
- **Languages:** English (configurable)
- **Speed:** Fast
- **Preprocessing:** No
- **Features:** Confidence scores, text detection

## 📦 Dependencies

```
Flask==3.0.0
Werkzeug==3.0.1
numpy==1.26.4
scipy==1.13.0
opencv-python==4.10.0.84
Pillow==10.3.0
pymupdf==1.24.0
pytesseract==0.3.10
easyocr==1.7.2
paddlepaddle==3.0.0
paddleocr==2.8.1
torch==2.2.0
torchvision==0.17.0
scikit-image==0.24.0
```

## 🐛 Troubleshooting

### Issue: "Tesseract not found"
**Solution:** Install Tesseract and add to system PATH

### Issue: PDF not processing
**Solution:** PyMuPDF is included - no additional setup needed

### Issue: EasyOCR taking long on first run
**Solution:** Normal behavior - downloading model files (happens once)

### Issue: Import errors
**Solution:** 
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Notes

- First run of EasyOCR downloads model files (~100MB)
- Extracted text is saved in `ocr_results/` folder
- Temporary images are cleaned up automatically
- Use forward slashes (`/`) in file paths, even on Windows

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License

## 👨‍💻 Author

Created with ❤️ by Sanjay

---

**Happy OCR Processing! 🎉**