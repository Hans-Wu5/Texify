# table_extractor.py

import cv2
import numpy as np
from .table_segmenter import segment_table_cells
from .table_handwriting_ocr import HandwritingOCR
from .table_to_latex import table_to_latex


ALLOWED = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-")


def clean_cell_text(t):
    """Remove garbage OCR characters."""
    return "".join([c for c in t if c in ALLOWED]).strip()


def pad_crop(img, pad=12):
    """Pad a cell crop before sending to OCR."""
    h, w = img.shape[:2]
    canvas = 255 * np.ones((h + pad*2, w + pad*2, 3), dtype=np.uint8)
    canvas[pad:pad+h, pad:pad+w] = img
    return canvas


def recognize_table(image_path):
    """
    1. Detect table region externally (content_detector)
    2. Segment rows and columns
    3. OCR each cell
    4. Convert to LaTeX
    """

    # -------------------------------
    # 1. Load and preprocess image
    # -------------------------------
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 2. Segment the table
    # -------------------------------
    cell_boxes = segment_table_cells(gray)

    if not cell_boxes:
        raise ValueError("Table segmentation failed — no cells detected.")

    # -------------------------------
    # 3. OCR setup
    # -------------------------------
    ocr = HandwritingOCR("microsoft/trocr-base-handwritten")

    table_tokens = []

    # -------------------------------
    # 4. OCR EACH CELL
    # -------------------------------
    for row in cell_boxes:
        row_tokens = []

        for (x, y, w, h) in row:
            crop = img[y:y+h, x:x+w]

            padded = pad_crop(crop)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

            raw_text = ocr.recognize(rgb)
            cleaned = clean_cell_text(raw_text)

            row_tokens.append(cleaned)

        table_tokens.append(row_tokens)

    # -------------------------------
    # 5. Convert to LaTeX
    # -------------------------------
    latex = table_to_latex(table_tokens)
    return latex


# --------------------------------
# CLI quick test
# --------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    print("\n===== TABLE OCR RESULT =====\n")
    print(recognize_table(path))