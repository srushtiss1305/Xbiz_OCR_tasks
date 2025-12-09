from paddleocr import PaddleOCR

# Initialize PaddleOCR (only once)
ocr = PaddleOCR(lang='en', use_angle_cls=True, show_log=False)

def process_image(img):
    """Extract text using PaddleOCR"""
    try:
        result = ocr.ocr(img, cls=True)
        
        if not result or result == [None]:
            return "(No text detected by PaddleOCR)"
        
        lines = []
        
        for page_result in result:
            if page_result is None:
                continue
            
            for line in page_result:
                if line is None:
                    continue
                
                if len(line) == 2:
                    bbox, text_info = line
                    
                    if isinstance(text_info, (tuple, list)) and len(text_info) == 2:
                        text = text_info[0]
                        conf = text_info[1]
                    elif isinstance(text_info, str):
                        text = text_info
                        conf = 1.0
                    else:
                        continue
                    
                    if conf >= 0.2:
                        lines.append(text)
        
        # Join with newlines
        text = '\n'.join(lines)
        return text if text else "(No text detected by PaddleOCR)"
    
    except Exception as e:
        return f"PaddleOCR Error: {str(e)}"