import easyocr

# Initialize EasyOCR reader (only once)
reader = easyocr.Reader(['en', 'mr'], gpu=False)

def process_image(img):
    """
    Process image using EasyOCR
    
    Args:
        img: numpy array of image (RGB)
    
    Returns:
        str: Extracted text
    """
    try:
        # Run EasyOCR
        results = reader.readtext(img, detail=0, paragraph=True)
        
        # Join all text results
        text = " ".join(results)
        
        return text.strip()
    
    except Exception as e:
        return f"EasyOCR Error: {str(e)}"