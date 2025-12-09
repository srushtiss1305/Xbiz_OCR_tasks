from flask import Flask, request, jsonify
import os, base64, uuid
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import io

# Import OCR modules
import tesseract_ocr
import easyocr_ocr
import paddle_ocr

app = Flask(__name__)
os.makedirs("images", exist_ok=True)
os.makedirs("ocr_results", exist_ok=True)

# OCR Engine mapping
OCR_ENGINES = {
    1: ("Tesseract", tesseract_ocr.process_image),
    2: ("EasyOCR", easyocr_ocr.process_image),
    3: ("PaddleOCR", paddle_ocr.process_image)
}

def is_base64(s):
    """Check if string is base64 encoded"""
    try:
        if len(s) < 50:  # Too short to be a valid image base64
            return False
        # Try to decode
        base64.b64decode(s, validate=True)
        return True
    except:
        return False

@app.route('/ocr', methods=['POST'])
def run_ocr():
    """
    Endpoint to run OCR on image from base64 or file path
    
    JSON body:
    {
        "image": "base64_string_OR_file_path",
        "ocr_id": 1,
        "documentType": "optional"
    }
    """
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get parameters
        image_input = data.get('image')
        ocr_id = data.get('ocr_id')
        doc = data.get('documentType', '')
        
        # Validate image input
        if not image_input:
            return jsonify({"error": "image field is required (base64 or file path)"}), 400
        
        # Validate OCR ID
        if not ocr_id:
            return jsonify({"error": "ocr_id is required (1=Tesseract, 2=EasyOCR, 3=PaddleOCR)"}), 400
        
        try:
            ocr_id = int(ocr_id)
        except ValueError:
            return jsonify({"error": "ocr_id must be a number (1, 2, or 3)"}), 400
        
        if ocr_id not in OCR_ENGINES:
            return jsonify({"error": f"Invalid ocr_id. Use 1=Tesseract, 2=EasyOCR, 3=PaddleOCR"}), 400
        
        # Check if input is base64 or file path
        is_base64_input = is_base64(image_input)
        temp_file_created = False
        
        if is_base64_input:
            # Input is base64 - decode it
            try:
                image_data = base64.b64decode(image_input)
                base64_image = image_input  # Already have base64
                
                # Save to temporary file to determine format
                img_id = str(uuid.uuid4())
                temp_path = f"images/{img_id}_temp"
                
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                
                temp_file_created = True
                
                # Try to detect file type
                try:
                    img = Image.open(io.BytesIO(image_data))
                    ext = f".{img.format.lower()}" if img.format else '.png'
                except:
                    ext = '.pdf'  # Assume PDF if not an image
                
                image_path = temp_path
                input_type = "base64"
                
            except Exception as e:
                return jsonify({"error": f"Failed to decode base64: {str(e)}"}), 400
        
        else:
            # Input is file path
            image_path = image_input
            
            # Check if file exists
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}"}), 400
            
            '''# Convert to base64
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode()
            except Exception as e:
                return jsonify({"error": f"Failed to read file: {str(e)}"}), 400'''
            
            ext = os.path.splitext(image_path)[1].lower()
            input_type = "file_path"
        
        # Convert to image array(s)
        try:
            if ext == '.pdf':
                # Use PyMuPDF to convert PDF to images
                pdf_document = fitz.open(image_path)
                imgs = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    # Render page to image (higher DPI = better quality)
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
        
        # Get selected OCR engine
        engine_name, ocr_function = OCR_ENGINES[ocr_id]
        
        # Process images with selected OCR
        extracted_text = ""
        for img in imgs:
            text = ocr_function(img)
            extracted_text += text + "\n"
        
        extracted_text = extracted_text.strip()
        
        # Save to .txt file
        result_id = str(uuid.uuid4())
        txt_filename = f"{engine_name.lower()}_{result_id}.txt"
        txt_path = f"ocr_results/{txt_filename}"
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        # Cleanup temp file if created
        if temp_file_created:
            os.remove(image_path)
        
        # Response
        return jsonify({
            "input_type": input_type,
            "ocr_engine": engine_name,
            "ocr_id": ocr_id,
            "extracted_text": extracted_text,
            "saved_file": txt_path,
            "msg": f"OCR completed using {engine_name}",
            "remark": "success"
        })
    
    except Exception as e:
        # Cleanup on error
        if temp_file_created and 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({
            "error": str(e),
            "remark": "failed"
        }), 500

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
    print("Accepts: Base64 encoded image OR file path")
    print("Results saved in: ocr_results/")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8080, debug=True)