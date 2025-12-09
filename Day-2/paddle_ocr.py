from paddleocr import PaddleOCR

# Initialize PaddleOCR (only once)
ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

def process_image(img):
    """
    Process image using PaddleOCR
    
    Args:
        img: numpy array of image (RGB)
    
    Returns:
        str: Extracted text
    """
    try:
        # Run PaddleOCR
        result = ocr.ocr(img, cls=True)
        
        lines = []
        
        if result and result != [None]:
            for page_result in result:
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
                        # Old/fallback format: text_info = "text only" (string)
                        elif isinstance(text_info, str):
                            text = text_info
                            conf = 0.0
                        else:
                            continue
                        
                        # Filter low confidence results (optional)
                        if conf == 0.0 or conf > 0.3:
                            lines.append(text)
        
        # Join all lines
        text = " ".join(lines)
        
        return text.strip() if text else "(No text detected by PaddleOCR)"
    
    except Exception as e:
        return f"PaddleOCR Error: {str(e)}"