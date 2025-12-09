import easyocr

# Initialize EasyOCR reader (only once)
reader = easyocr.Reader(['en', 'mr'], gpu=False)

def process_image(img):
    """Extract text using EasyOCR"""
    try:
        results = reader.readtext(img, detail=0, paragraph=True)
        
        if not results:
            return "(No text detected by EasyOCR)"
        
        # Join with newlines
        text = '\n'.join(results)
        return text
    
    except Exception as e:
        return f"EasyOCR Error: {str(e)}"