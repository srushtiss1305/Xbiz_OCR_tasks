from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance
import os
import fitz  # PyMuPDF
from datetime import datetime
from typing import Union
from pathlib import Path
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
IMAGE_PATH = "input_image.jpg"  # For /process endpoint
OUTPUT_FOLDER = "processed_images"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Create folders if they don't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

def allowed_file(image_path):
    if '.' in image_path and image_path.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        ext = os.path.splitext(image_path)[1].lower()
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
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400


def check_and_correct_skew(image_path):
    """Check if image needs deskewing and apply if needed"""
    print("Step 1: Checking skew...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            
            # Only deskew if angle is significant (> 0.5 degrees)
            if abs(median_angle) > 0.5:
                print(f"  Skew detected: {median_angle:.2f} degrees - CORRECTING")
                
                # Get image dimensions
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                
                # Perform rotation
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                corrected = cv2.warpAffine(img, M, (w, h), 
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
                
                output_path = os.path.join(OUTPUT_FOLDER, "1_deskewed.jpg")
                cv2.imwrite(output_path, corrected)
                print(f"  ✓ Saved: {output_path}")
                return output_path
    
    print("  No significant skew detected")
    output_path = os.path.join(OUTPUT_FOLDER, "1_deskewed.jpg")
    cv2.imwrite(output_path, img)
    return output_path


def detect_rotation_angle(image_path):
    """Detect if image needs 90/180/270 degree rotation with robust logic"""
    print("Step 2: Checking rotation...")
    
    img = cv2.imread(image_path)
    
    # Try all 4 orientations and get OCR results
    orientations = [0, 90, 180, 270]
    orientation_scores = {}
    
    for angle in orientations:
        # Rotate image
        if angle == 0:
            rotated = img.copy()
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:  # 270
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Save temporarily
        temp_path = os.path.join(OUTPUT_FOLDER, f"temp_rotation_{angle}.jpg")
        cv2.imwrite(temp_path, rotated)
        
        # Run OCR to get confidence and text count
        try:
            result = ocr.predict(temp_path)
            total_confidence = 0
            text_count = 0
            high_confidence_count = 0  # >0.7
            very_high_confidence_count = 0  # >0.9
            ultra_high_confidence_count = 0  # >0.95
            confidence_list = []
            
            if result and len(result) > 0:
                for page_result in result:
                    if isinstance(page_result, dict):
                        rec_texts = page_result.get('rec_texts', [])
                        rec_scores = page_result.get('rec_scores', [])
                        if rec_scores:
                            confidence_list = list(rec_scores)
                            total_confidence = sum(rec_scores)
                            text_count = len(rec_scores)
                            high_confidence_count = sum(1 for s in rec_scores if s > 0.7)
                            very_high_confidence_count = sum(1 for s in rec_scores if s > 0.9)
                            ultra_high_confidence_count = sum(1 for s in rec_scores if s > 0.95)
            
            avg_confidence = total_confidence / text_count if text_count > 0 else 0
            
            # Enhanced weighted score
            base_score = avg_confidence * 50  # Base confidence (0-50 points)
            text_count_score = min(text_count * 0.5, 20)  # Text count (0-20 points)
            high_conf_score = high_confidence_count * 1.5  # High conf bonus
            very_high_conf_score = very_high_confidence_count * 3  # Very high conf bonus
            ultra_high_conf_score = ultra_high_confidence_count * 5  # Ultra high conf bonus
            
            combined_score = (base_score + text_count_score + high_conf_score + 
                            very_high_conf_score + ultra_high_conf_score)
            
            # Consistency bonus
            if confidence_list and len(confidence_list) > 3:
                conf_std = np.std(confidence_list)
                if conf_std < 0.15:  # Very consistent
                    combined_score *= 1.1
            else:
                conf_std = 1.0
            
            orientation_scores[angle] = {
                'avg_confidence': avg_confidence,
                'text_count': text_count,
                'high_conf_count': high_confidence_count,
                'very_high_conf_count': very_high_confidence_count,
                'ultra_high_conf_count': ultra_high_confidence_count,
                'conf_std': conf_std if confidence_list else 1.0,
                'combined_score': combined_score,
                'confidence_list': confidence_list,
                'total_confidence': total_confidence
            }
            
            print(f"  Angle {angle:3d}°: avg_conf={avg_confidence:.3f}, texts={text_count:3d}, "
                  f"high={high_confidence_count:2d}, v_high={very_high_confidence_count:2d}, "
                  f"score={combined_score:.1f}")
            
        except Exception as e:
            print(f"  Error testing angle {angle}: {e}")
            orientation_scores[angle] = {
                'avg_confidence': 0,
                'text_count': 0,
                'high_conf_count': 0,
                'very_high_conf_count': 0,
                'ultra_high_conf_count': 0,
                'conf_std': 1.0,
                'combined_score': 0,
                'confidence_list': [],
                'total_confidence': 0
            }
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Find best angle
    best_angle = max(orientation_scores, key=lambda k: orientation_scores[k]['combined_score'])
    best_score = orientation_scores[best_angle]['combined_score']
    zero_score = orientation_scores[0]['combined_score']
    
    # Get second best for comparison
    sorted_angles = sorted(orientation_scores.items(), 
                          key=lambda x: x[1]['combined_score'], 
                          reverse=True)
    second_best_angle = sorted_angles[1][0] if len(sorted_angles) > 1 else 0
    second_best_score = sorted_angles[1][1]['combined_score'] if len(sorted_angles) > 1 else 0
    
    print(f"\n  Ranking: 1st={best_angle}° ({best_score:.1f}), "
          f"2nd={second_best_angle}° ({second_best_score:.1f})")
    print(f"  0° score: {zero_score:.1f}")
    
    # Decision logic with multiple criteria
    should_rotate = False
    rotation_reason = ""
    
    if best_angle == 0:
        rotation_reason = "0° is already the best orientation"
    else:
        # Calculate metrics
        score_diff = best_score - zero_score
        score_ratio = best_score / zero_score if zero_score > 0 else float('inf')
        
        # Criterion 1: 0° score is very poor (likely wrong orientation)
        if zero_score < 10:
            should_rotate = True
            rotation_reason = f"0° has very poor score ({zero_score:.1f})"
        
        # Criterion 2: Best angle score is significantly better (2x or 50+ points better)
        elif score_ratio >= 2.0:
            should_rotate = True
            rotation_reason = f"best angle {score_ratio:.1f}x better than 0°"
        
        # Criterion 3: Absolute score difference is large
        elif score_diff >= 30:
            should_rotate = True
            rotation_reason = f"score difference of {score_diff:.1f} points"
        
        # Criterion 4: Best angle has many high-confidence detections, 0° doesn't
        elif (orientation_scores[best_angle]['very_high_conf_count'] >= 3 and 
              orientation_scores[0]['very_high_conf_count'] == 0):
            should_rotate = True
            rotation_reason = "best angle has high-confidence text, 0° doesn't"
        
        # Criterion 5: Moderate improvement (1.5x) with reasonable text count
        elif score_ratio >= 1.5 and orientation_scores[best_angle]['text_count'] >= 5:
            should_rotate = True
            rotation_reason = f"1.5x+ improvement with good text detection"
        
        # Criterion 6: Best angle has much better average confidence
        elif (orientation_scores[best_angle]['avg_confidence'] > 0.75 and 
              orientation_scores[0]['avg_confidence'] < 0.5):
            should_rotate = True
            rotation_reason = f"much better avg confidence ({orientation_scores[best_angle]['avg_confidence']:.2f} vs {orientation_scores[0]['avg_confidence']:.2f})"
        
        # Criterion 7: Clear winner with gap to second place
        elif score_diff >= 20 and (best_score - second_best_score) >= 15:
            should_rotate = True
            rotation_reason = f"clear winner with {score_diff:.1f} point lead"
    
    if should_rotate:
        print(f"  ✓ ROTATING to {best_angle}°: {rotation_reason}")
    else:
        if best_angle != 0:
            print(f"  ✗ NO ROTATION: {rotation_reason if rotation_reason else 'improvement not significant enough'}")
            print(f"     (Best: {best_angle}° with {best_score:.1f}, but keeping 0° with {zero_score:.1f})")
        else:
            print(f"  ✓ NO ROTATION NEEDED: {rotation_reason}")
    
    # Apply rotation if needed
    if should_rotate:
        if best_angle == 90:
            corrected = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif best_angle == 180:
            corrected = cv2.rotate(img, cv2.ROTATE_180)
        elif best_angle == 270:
            corrected = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            corrected = img
    else:
        corrected = img
    
    output_path = os.path.join(OUTPUT_FOLDER, "2_rotated.jpg")
    cv2.imwrite(output_path, corrected)
    print(f"  ✓ Saved: {output_path}")
    return output_path 

def preprocess_image(image_path):
    """Apply comprehensive preprocessing for better OCR"""
    print("Step 3: Preprocessing image...")
    
    # Read image
    img = cv2.imread(image_path)
    
    # 1. Convert to PIL for better quality operations
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 2. Upscale 3x for better OCR (increased from 2x)
    width, height = pil_img.size
    pil_img = pil_img.resize((width * 3, height * 3), Image.LANCZOS)
    print(f"  ✓ Upscaled to {width*3}x{height*3}")
    
    # 3. Enhance sharpness moderately
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.8)
    print("  ✓ Sharpness enhanced")
    
    # 4. Enhance contrast moderately
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.5)
    print("  ✓ Contrast enhanced")
    
    # Convert back to OpenCV
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 5. Denoise very gently
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    print("  ✓ Denoising applied")
    
    # 6. Convert to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    # 7. Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    print("  ✓ CLAHE applied")
    
    # 8. Slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    print("  ✓ Gaussian blur applied")
    
    # Save the grayscale version - PaddleOCR works well with clean grayscale
    output_path = os.path.join(OUTPUT_FOLDER, "3_preprocessed.jpg")
    cv2.imwrite(output_path, blurred)
    print(f"  ✓ Saved: {output_path}")
    
    return output_path


def extract_text_with_ocr(image_path):
    """Extract text using PaddleOCR"""
    print("Step 4: Extracting text with PaddleOCR...")
    
    try:
        # Initialize OCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        
        print(f"  Reading image: {image_path}")
        
        # Run OCR
        result = ocr.predict(image_path)
        
        print(f"  Result type: {type(result)}")
        print(f"  Result length: {len(result)}")
        
        if not result or len(result) == 0:
            print("  ⚠ No text detected by OCR")
            return ""
        
        # Extract text from the new PaddleOCR result format
        extracted_lines = []
        
        # The result is a list with one OCRResult object (dictionary)
        for idx, page_result in enumerate(result):
            print(f"  Page {idx} type: {type(page_result)}")
            print(f"  Is dict: {isinstance(page_result, dict)}")
            print(f"  Has rec_texts attr: {hasattr(page_result, 'rec_texts')}")
            
            # Check if it's a dictionary with 'rec_texts' and 'rec_scores'
            if isinstance(page_result, dict):
                print(f"  Dictionary keys: {list(page_result.keys())}")
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                
                print(f"  rec_texts found: {len(rec_texts)} items")
                print(f"  rec_texts content: {rec_texts}")
                
                if rec_texts:
                    print(f"  Found {len(rec_texts)} text regions:")
                    
                    for text_idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                        extracted_lines.append(text)
                        print(f"    {text_idx+1}. '{text}' (conf: {score:.2f})")
            
            # Fallback: check if it has these as attributes
            elif hasattr(page_result, 'rec_texts'):
                rec_texts = page_result.rec_texts
                rec_scores = page_result.rec_scores
                
                print(f"  Found {len(rec_texts)} text regions (via attributes):")
                
                for text_idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                    extracted_lines.append(text)
                    print(f"    {text_idx+1}. '{text}' (conf: {score:.2f})")
            
            # Try accessing as object properties
            else:
                print(f"  Trying to access as object...")
                try:
                    if hasattr(page_result, '__dict__'):
                        print(f"  Object dict: {page_result.__dict__.keys()}")
                except:
                    pass
        
        print(f"  Total extracted lines: {len(extracted_lines)}")
        
        if not extracted_lines:
            print("  ⚠ No text could be extracted from OCR results")
            return ""
        
        full_text = "\n".join(extracted_lines)
        print(f"  ✓ Successfully extracted {len(extracted_lines)} lines of text")
        
        return full_text
        
    except Exception as e:
        print(f"  ✗ OCR Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def postprocess_text(text):
    """Enhanced text cleanup and formatting"""
    if not text or text.strip() == "":
        return text
    
    print("Step 5: Postprocessing text...")
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove confidence annotations
        line = re.sub(r'\(conf(?:idence)?:\s*[\d.]+\)', '', line)
        line = re.sub(r'\[[\d.]+\]', '', line)  # Remove [0.95] style annotations
        
        # Remove extra whitespace
        line = ' '.join(line.split())
        
        # Fix common OCR errors
        # Fix common character confusions
        replacements = {
            '|': 'I',  # Pipe to I
            '0': 'O',  # Zero to O in words (context dependent)
            '§': 'S',
            '©': 'C',
            '®': 'R',
        }
        
        # Apply replacements carefully (only in certain contexts)
        # Skip if line looks like a number/code
        if not re.match(r'^[\d\s\-./]+$', line):
            for old, new in replacements.items():
                # Only replace if it looks like it's in a word
                line = re.sub(f'(?<=[a-zA-Z]){re.escape(old)}(?=[a-zA-Z])', new, line)
        
        # Fix spacing around punctuation
        line = re.sub(r'\s+([.,!?;:])', r'\1', line)
        line = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', line)
        
        # Remove lines that are just special characters or very short noise
        if len(line.strip()) < 2:
            continue
        
        # Skip lines with too many special characters (likely OCR noise)
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / max(len(line), 1)
        if special_char_ratio > 0.5 and len(line) < 10:
            continue
        
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    # Join lines intelligently
    # If a line doesn't end with punctuation and next line doesn't start with capital,
    # they might belong together
    merged_lines = []
    i = 0
    while i < len(cleaned_lines):
        current = cleaned_lines[i]
        
        # Check if we should merge with next line
        if i < len(cleaned_lines) - 1:
            next_line = cleaned_lines[i + 1]
            
            # Merge conditions:
            # 1. Current doesn't end with sentence-ending punctuation
            # 2. Next doesn't start with capital letter
            # 3. Both are reasonably long (not just fragments)
            should_merge = (
                not re.search(r'[.!?]$', current) and
                len(next_line) > 0 and
                not next_line[0].isupper() and
                len(current) > 3 and
                len(next_line) > 3
            )
            
            if should_merge:
                merged_lines.append(current + ' ' + next_line)
                i += 2
                continue
        
        merged_lines.append(current)
        i += 1
    
    processed = '\n'.join(merged_lines)
    
    # Final cleanup
    processed = re.sub(r'\n{3,}', '\n\n', processed)  # Max 2 consecutive newlines
    processed = processed.strip()
    
    print(f"  ✓ Cleaned and formatted {len(merged_lines)} lines")
    
    return processed


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process image endpoint"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING UPLOADED FILE: {unique_filename}")
        print(f"{'='*60}\n")
        
        # Process the image
        deskewed_path = check_and_correct_skew(file_path)
        rotated_path = detect_rotation_angle(deskewed_path)
        preprocessed_path = preprocess_image(rotated_path)
        extracted_text = extract_text_with_ocr(preprocessed_path)
        final_text = postprocess_text(extracted_text)
        
        # Save text
        text_filename = f"extracted_text_{timestamp}.txt"
        text_path = os.path.join(OUTPUT_FOLDER, text_filename)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"\n{'='*60}")
        print(f"✓ UPLOAD PROCESSING COMPLETE")
        print(f"  Text file: {text_path}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'filename': unique_filename,
            'output_folder': OUTPUT_FOLDER,
            'files': {
                'deskewed': '1_deskewed.jpg',
                'rotated': '2_rotated.jpg',
                'preprocessed': '3_preprocessed.jpg',
                'text': text_filename
            },
            'extracted_text': final_text,
            'line_count': len(final_text.split('\n')) if final_text else 0
        }), 200
        
    except Exception as e:
        print(f"\n✗ UPLOAD ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/process', methods=['GET'])
def process_image():
    """Main endpoint to process the image"""
    try:
        if not os.path.exists(IMAGE_PATH):
            return jsonify({
                'error': f'Image not found at {IMAGE_PATH}'
            }), 404
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {IMAGE_PATH}")
        print(f"{'='*60}\n")
        
        # Step 1: Skew correction
        deskewed_path = check_and_correct_skew(IMAGE_PATH)
        
        # Step 2: Rotation correction
        rotated_path = detect_rotation_angle(deskewed_path)
        
        # Step 3: Preprocess
        preprocessed_path = preprocess_image(rotated_path)
        
        # Step 4: Extract text
        extracted_text = extract_text_with_ocr(preprocessed_path)
        
        # Step 5: Postprocess
        final_text = postprocess_text(extracted_text)
        
        # Save text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_filename = f"extracted_text_{timestamp}.txt"
        text_path = os.path.join(OUTPUT_FOLDER, text_filename)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        print(f"\n{'='*60}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"  Text file: {text_path}")
        print(f"  Total lines: {len(final_text.split(chr(10)))}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            'output_folder': OUTPUT_FOLDER,
            'files': {
                'deskewed': '1_deskewed.jpg',
                'rotated': '2_rotated.jpg',
                'preprocessed': '3_preprocessed.jpg',
                'text': text_filename
            },
            'extracted_text': final_text,
            'line_count': len(final_text.split('\n')) if final_text else 0
        }), 200
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Flask OCR API Starting")
    print(f"{'='*60}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"\nEndpoint: http://localhost:8082/process")
    print(f"{'='*60}\n")
    app.run(debug=True, port=8082)