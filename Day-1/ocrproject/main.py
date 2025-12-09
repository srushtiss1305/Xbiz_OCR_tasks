# main.py
from flask import Flask, request, jsonify
import os, base64, uuid, cv2, numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import easyocr
from paddleocr import PaddleOCR

app = Flask(__name__)
os.makedirs("images", exist_ok=True)
os.makedirs("ocr_results", exist_ok=True)  # Folder to save .txt files

# Initialize OCR engines
reader = easyocr.Reader(['en', 'mr'], gpu=False)
ocr = PaddleOCR(lang='en', device='cpu')  # Clean, no warnings

def best_preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.shape[1] < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8)).apply(gray)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return gray

@app.route('/ocr', methods=['POST'])
def run():
    txn = request.form.get('txn_id', str(uuid.uuid4()))
    doc = request.form.get('documentType', '')

    file = request.files.get('file')
    if not file or not file.filename:
        return jsonify({"error": "no file"}), 400

    # Save uploaded image
    img_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    img_path = f"images/{img_id}{ext}"
    file.save(img_path)

    # Convert to image array(s)
    if ext == '.pdf':
        pages = convert_from_path(img_path, dpi=300)
        imgs = [np.array(p.convert('RGB')) for p in pages]
    else:
        imgs = [np.array(Image.open(img_path).convert('RGB'))]

    # Encode original image
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    tess_text = easy_text = paddle_text = ""

    for img in imgs:
        pre = best_preprocess(img)

        # Tesseract
        tess_text += pytesseract.image_to_string(pre, lang='eng+hin+mar', config='--oem 3 --psm 6') + "\n"

        # EasyOCR
        easy_text += " ".join(reader.readtext(img, detail=0, paragraph=True)) + "\n"

        # PaddleOCR
                # PaddleOCR - FIXED & WORKING
        paddle_lines = []
        try:
            result = ocr.predict(img)
            print(result[0]['rec_texts'])
            lines = []
            if result and result != [None]:
                for page_result in result:               # usually only 1 page
                    if page_result is None:
                        continue
                    for line in page_result:
                        if line is None:
                            continue

                        # Handle both formats safely
                        if len(line) == 2:
                            bbox, text_info = line

                            # New format: text_info = (text, confidence)
                            if isinstance(text_info, (tuple, list)) and len(text_info) == 2:
                                text = text_info[0]
                                conf = text_info[1]
                            # Old or fallback format: text_info = "text only" (string)
                            elif isinstance(text_info, str):
                                text = text_info
                                conf = 0.0
                            else:
                                continue

                            # Optional: filter very low confidence
                            if conf == 0.0 or conf > 0.3:
                                lines.append(text)

            page_text = " ".join(lines).strip()
            paddle_text += page_text + "\n" if page_text else "(No text detected by PaddleOCR)\n"

        except Exception as e:
            paddle_text += f"(PaddleOCR crashed: {str(e)})\n"

    # Save to .txt files
    tess_file = f"ocr_results/tesseract.txt"
    easy_file = f"ocr_results/easyocr.txt"
    paddle_file = f"ocr_results/paddleocr.txt"

    with open(tess_file, "w", encoding="utf-8") as f:
        f.write(tess_text.strip())
    with open(easy_file, "w", encoding="utf-8") as f:
        f.write(easy_text.strip())
    with open(paddle_file, "w", encoding="utf-8") as f:
        f.write(" ".join(result[0]['rec_texts']).strip())

    # Cleanup uploaded image
    os.remove(img_path)

    # Response
    return jsonify({
        "txn_id": txn,
        "documentType": doc,
        "tesseract": {
            "text": tess_text.strip(),
            "saved_file": tess_file
        },
        "easyocr": {
            "text": easy_text.strip(),
            "saved_file": easy_file
        },
        "paddleocr": {
            "text": " ".join(result[0]['rec_texts']).strip(),
            "saved_file": paddle_file
        },
        "msg": "OCR completed and saved to .txt files",
        "remark": "success"
    })

if __name__ == '__main__':
    print("OCR API Running → http://127.0.0.1:8081/ocr")
    print("Results saved in: ocr_results/")
    app.run(host='0.0.0.0', port=8081)