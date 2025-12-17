from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import os
import fitz  # PyMuPDF
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
import re
from threading import Lock
import traceback

# -------------------- ENV SAFETY --------------------
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
os.environ["OMP_NUM_THREADS"] = "1"

# -------------------- APP INIT --------------------
app = Flask(__name__)
CORS(app)

# -------------------- PATHS --------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "processed_images")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

# -------------------- OCR INIT --------------------
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    text_det_limit_side_len=960
)

# PaddleOCR is NOT thread-safe
ocr_lock = Lock()

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_transaction_id():
    return f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def safe_imread(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def pdf_to_images(pdf_path, txn_id):
    images = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        # Reduced DPI to avoid memory crashes
        pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        out_path = os.path.join(
            OUTPUT_FOLDER, f"{txn_id}_page_{i}.jpg"
        )
        cv2.imwrite(out_path, np.array(img))
        images.append(out_path)

    doc.close()
    return images


def deskew_image(image_path, txn_id):
    img = safe_imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            deg = np.degrees(theta) - 90
            if -45 < deg < 45:
                angles.append(deg)
        if angles:
            angle = np.median(angles)

    if abs(angle) > 0.5:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    out = os.path.join(
        OUTPUT_FOLDER, f"{txn_id}_deskewed.jpg"
    )
    cv2.imwrite(out, img)
    return out


def preprocess_image(image_path, txn_id):
    img = safe_imread(image_path)

    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = os.path.join(
        OUTPUT_FOLDER, f"{txn_id}_preprocessed.jpg"
    )
    cv2.imwrite(out, gray)
    return out


def run_ocr(image_path):
    # PaddleOCR must be locked
    with ocr_lock:
        result = ocr.predict(image_path)

    lines = []
    for page in result:
        if isinstance(page, dict):
            lines.extend(page.get("rec_texts", []))

    return "\n".join(lines)


def clean_text(text):
    if not text:
        return ""

    cleaned = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) > 1:
            cleaned.append(line)

    return "\n".join(cleaned)


# -------------------- ROUTES --------------------
@app.route("/upload", methods=["POST"])
def upload():
    txn_id = generate_transaction_id()

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        input_path = os.path.join(
            UPLOAD_FOLDER, f"{txn_id}_{filename}"
        )
        file.save(input_path)

        extracted_texts = []

        if input_path.lower().endswith(".pdf"):
            pages = pdf_to_images(input_path, txn_id)
            for page_path in pages:
                deskewed = deskew_image(page_path, txn_id)
                pre = preprocess_image(deskewed, txn_id)
                extracted_texts.append(run_ocr(pre))
        else:
            deskewed = deskew_image(input_path, txn_id)
            pre = preprocess_image(deskewed, txn_id)
            extracted_texts.append(run_ocr(pre))

        final_text = clean_text("\n\n".join(extracted_texts))

        text_path = os.path.join(
            OUTPUT_FOLDER, f"{txn_id}_extracted.txt"
        )
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return jsonify({
            "status": "success",
            "transaction_id": txn_id,
            "text_file": text_path,
            "extracted_text": final_text
        })

    except Exception as e:
        print(f"[{txn_id}] ERROR")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "transaction_id": txn_id,
            "message": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------- GLOBAL ERROR HANDLER --------------------
@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({
        "status": "error",
        "message": str(e)
    }), 500


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("Flask PaddleOCR API running")
    app.run(host="0.0.0.0", port=9000, threaded=False)