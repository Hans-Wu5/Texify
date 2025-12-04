import sys
import cv2
from PIL import Image
from backend.handwriting_ocr import HandwritingOCR

def run_raw_ocr(image_path):
    print("[INFO] Loading image…")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read image")

    # Convert to PIL RGB for TrOCR
    pil = Image.fromarray(img).convert("RGB")

    print("[INFO] Running TrOCR on raw image…")
    ocr = HandwritingOCR("microsoft/trocr-base-handwritten")
    txt = ocr.recognize(pil)

    print("\n===== RAW OCR OUTPUT =====\n")
    print(txt)
    print("\n==========================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_raw_ocr.py <path_to_image>")
        sys.exit(1)

    run_raw_ocr(sys.argv[1])