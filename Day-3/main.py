from flask import Flask, request, jsonify
import os, base64, uuid
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import io
import json
from datetime import datetime
import re

# Import OCR modules
import tesseract_ocr
import easyocr_ocr
import paddle_ocr

app = Flask(__name__)
os.makedirs("images", exist_ok=True)
os.makedirs("ocr_results", exist_ok=True)
os.makedirs("transactions", exist_ok=True)

# OCR Engine mapping
OCR_ENGINES = {
    1: ("Tesseract", tesseract_ocr.process_image),
    2: ("EasyOCR", easyocr_ocr.process_image),
    3: ("PaddleOCR", paddle_ocr.process_image)
}

def is_base64(s):
    """Check if string is base64 encoded"""
    try:
        if len(s) < 50:
            return False
        base64.b64decode(s, validate=True)
        return True
    except:
        return False

def clean_extracted_text(text):
    """
    Clean and format extracted text
    - Remove garbage characters
    - Preserve line structure
    - Remove extra spaces
    """
    # Remove special characters but keep alphanumeric, spaces, and common punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\@\#\$\%\&\*\+\=\/\\\'\"]+', '', text)
    
    # Remove multiple spaces but preserve single spaces
    text = re.sub(r' +', ' ', text)
    
    # Split by common separators and remove empty lines
    # Replace multiple newlines with single newline
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Add line break after certain punctuation
    #text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    # Process line by line
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Keep line if it has at least 2 alphanumeric characters
        if len(re.findall(r'[a-zA-Z0-9]', line)) >= 2:
            cleaned_lines.append(line)
    
    # Join lines with newline
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text.strip()

@app.route('/ocr', methods=['POST'])
def run_ocr():
    """
    Endpoint to run OCR on image from base64 or file path
    
    JSON body:
    {
        "image": "base64_string_OR_file_path",
        "ocr_id": 1,
        "txn_id": "optional_transaction_id",
        "documentType": "optional"
    }
    """
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get parameters
        image_input = data.get('image')
        ocr_id = data.get('ocr_id')
        doc = data.get('documentType', '')
        txn_id = data.get('txn_id', str(uuid.uuid4()))
        
        # Create transaction folder
        if not txn_id:
            txn_folder = f"transactions/{txn_id}"
        else:
            txn_folder = f"transactions/{txn_id}_{str(uuid.uuid4())}"
        os.makedirs(txn_folder, exist_ok=True)
        
        # Validate inputs
        if not image_input:
            return jsonify({"error": "image field is required"}), 400
        
        if not ocr_id:
            return jsonify({"error": "ocr_id is required (1=Tesseract, 2=EasyOCR, 3=PaddleOCR)"}), 400
        
        try:
            ocr_id = int(ocr_id)
        except ValueError:
            return jsonify({"error": "ocr_id must be a number"}), 400
        
        if ocr_id not in OCR_ENGINES:
            return jsonify({"error": "Invalid ocr_id. Use 1, 2, or 3"}), 400
        
        # Check if input is base64 or file path
        is_base64_input = is_base64(image_input)
        temp_file_created = False
        
        if is_base64_input:
            # Decode base64
            try:
                image_data = base64.b64decode(image_input)
                base64_image = image_input
                
                img_id = str(uuid.uuid4())
                temp_path = f"images/{img_id}_temp"
                
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                
                temp_file_created = True
                
                try:
                    img = Image.open(io.BytesIO(image_data))
                    ext = f".{img.format.lower()}" if img.format else '.png'
                except:
                    ext = '.pdf'
                
                image_path = temp_path
                input_type = "base64"
                
            except Exception as e:
                return jsonify({"error": f"Failed to decode base64: {str(e)}"}), 400
        
        else:
            # File path
            image_path = image_input
            
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}"}), 400
            
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode()
            except Exception as e:
                return jsonify({"error": f"Failed to read file: {str(e)}"}), 400
            
            ext = os.path.splitext(image_path)[1].lower()
            input_type = "file_path"
        
        # Convert to image arrays
        try:
            if ext == '.pdf':
                pdf_document = fitz.open(image_path)
                imgs = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    imgs.append(np.array(img))
                pdf_document.close()
            else:
                img = Image.open(image_path)
                imgs = [np.array(img.convert('RGB'))]
        except Exception as e:
            if temp_file_created:
                os.remove(image_path)
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
        
        # Get OCR engine
        engine_name, ocr_function = OCR_ENGINES[ocr_id]
        
        # Process images
        extracted_text = ""
        for img in imgs:
            text = ocr_function(img)
            extracted_text += text + "\n"
        
        # Clean the extracted text (simple post-processing with \n)
        extracted_text = clean_extracted_text(extracted_text)
        
        # Save output.txt
        txt_path = f"{txn_folder}/output.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        # Save request.json
        request_data = {
            "txn_id": txn_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_input_type": input_type,
            "ocr_id": ocr_id,
            "ocr_engine": engine_name,
        }
        
        request_path = f"{txn_folder}/request.json"
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2)
        
        # Prepare response
        response_data = {
            "txn_id": txn_id,
            "input_type": input_type,
            "ocr_engine": engine_name,
            "ocr_id": ocr_id,
            "extracted_text": extracted_text,
            "saved_files": {
                "output": txt_path,
                "request": request_path,
                "response": f"{txn_folder}/response.json"
            },
            "msg": f"OCR completed using {engine_name}",
            "remark": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save response.json
        response_path = f"{txn_folder}/response.json"
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)
        
        # Cleanup
        if temp_file_created:
            os.remove(image_path)
        
        return jsonify(response_data)
    
    except Exception as e:
        if temp_file_created and 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": str(e), "remark": "failed"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "available_engines": {
            "1": "Tesseract",
            "2": "EasyOCR", 
            "3": "PaddleOCR"
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("OCR API Running → http://127.0.0.1:8080/ocr")
    print("=" * 50)
    print("Available OCR Engines:")
    print("  1 = Tesseract")
    print("  2 = EasyOCR")
    print("  3 = PaddleOCR")
    print("=" * 50)
    print("Accepts: Base64 OR file path")
    print("Transactions: transactions/[txn_id]/")
    print("  - request.json")
    print("  - response.json")
    print("  - output.txt")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8080, debug=True)