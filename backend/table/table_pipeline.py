# table_pipeline.py

import cv2
import numpy as np

from .table_handwriting_ocr import TableOCR
from ..content_detector import detect_content
from ..table.table_segmenter import segment_table_cells
from .table_to_latex import table_cells_to_latex
from pathlib import Path

DESKTOP = str(Path.home() / "Desktop")

# ------------------------------
# TEXT CLEANUP
# ------------------------------
ALLOWED = set("0123456789abcdefghijklmnopqrstuvwxyz"
              "ABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=().,:% ")


def clean_cell_text(txt):
    """Remove garbage characters from OCR output."""
    return "".join([c for c in txt if c in ALLOWED]).strip()


def prepare_for_trocr(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    return rgb


# ------------------------------
# MAIN TABLE RECOGNITION
# ------------------------------
def recognize_table(image_path):
    """Given an image file path → return LaTeX tabular."""

    # 1. Detect content
    orig, detections = detect_content(image_path)

    # Expecting ("table", bbox, …)
    if not detections or detections[0][0] != "table":
        raise ValueError("No table detected.")

    (_, (x1, y1, x2, y2)) = detections[0]

    # 2. Crop out the table region
    table_crop = orig[y1:y2, x1:x2]
    gray = cv2.cvtColor(table_crop, cv2.COLOR_BGR2GRAY)

    # 3. Segment into rows × columns
    cell_boxes = segment_table_cells(gray)
    if not cell_boxes:
        raise ValueError("Table segmentation failed")

    # 4. Load OCR
    ocr = TableOCR()

    # 5. OCR each cell
    table_tokens = []
    for r, row in enumerate(cell_boxes):
        row_tokens = []
        for c, (cx, cy, w, h) in enumerate(row):

            crop = table_crop[cy:cy + h, cx:cx + w]
            trocr_input = prepare_for_trocr(crop)

            # -------------------------------------------------
            # NEW: Save raw image fed into OCR
            # -------------------------------------------------
            debug_path = f"{DESKTOP}/table_ocr_cell_{r}_{c}.png"
            cv2.imwrite(debug_path, crop)
            print(f"[DEBUG] Saved OCR crop → {debug_path}")
            # -------------------------------------------------

            raw_text = ocr.recognize(trocr_input)
            cleaned = clean_cell_text(raw_text)
            row_tokens.append(cleaned)

        table_tokens.append(row_tokens)

    # 6. Convert table structure → LaTeX tabular
    latex_code = table_cells_to_latex(table_tokens)
    return latex_code


# ------------------------------
# CLI for quick testing
# ------------------------------
if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    latex = recognize_table(img_path)
    print("\n===== LaTeX TABLE =====\n")
    print(latex)