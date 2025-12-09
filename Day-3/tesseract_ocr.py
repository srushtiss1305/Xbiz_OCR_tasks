import cv2
import numpy as np
import pytesseract

def preprocess_image(img):
    """Preprocess image for better OCR results"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if gray.shape[1] < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(gray)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    return gray

def process_image(img):
    """Extract text using Tesseract OCR"""
    try:
        preprocessed = preprocess_image(img)
        text = pytesseract.image_to_string(
            preprocessed, 
            lang='eng+hin+mar',
            config='--oem 3 --psm 6'
        )
        return text
    except Exception as e:
        return f"Tesseract Error: {str(e)}"