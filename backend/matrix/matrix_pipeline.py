# matrix_pipeline.py

import cv2
from Texify.backend.content_detector import detect_content
from .matrix_segmenter import segment_matrix_cells
from Texify.backend.handwriting_ocr import HandwritingOCR
from .matrix_to_latex import tokens_to_matrix_latex

# ------------------------------
# NUMERIC CLEANUP
# ------------------------------
ALLOWED = set("0123456789.")

def clean_numeric_prediction(txt):
    """Remove non-numeric characters from OCR output."""
    return "".join([c for c in txt if c in ALLOWED]).strip()

# ------------------------------
# COLLAPSE EXTRA COLUMNS
# ------------------------------
def collapse_columns(cell_tokens):
    """
    Auto-detect rows with extra columns and collapse them so *all rows*
    have the same number of columns (the global minimum).
    """
    # Column count per row
    col_counts = [len(row) for row in cell_tokens]
    min_cols = min(col_counts)

    collapsed = []
    for row in cell_tokens:
        if len(row) == min_cols:
            collapsed.append(row)
        else:
            # Merge all columns into min_cols cells (just concatenation)
            merged = ["".join(row[:len(row)])]
            collapsed.append(merged)
    return collapsed


def recognize_matrix(image_path):
    # 1. Detect content
    orig, detections = detect_content(image_path)

    if not detections or detections[0][0] != "matrix":
        raise ValueError("No matrix detected")

    (_, (x1, y1, x2, y2), left_b, right_b) = detections[0]

    # 2. Extract matrix crop
    matrix_crop = orig[y1:y2, x1:x2]
    gray = cv2.cvtColor(matrix_crop, cv2.COLOR_BGR2GRAY)

    # 3. Segment into cells
    cell_boxes = segment_matrix_cells(gray, left_b, right_b)
    if not cell_boxes:
        raise ValueError("Matrix segmentation failed")

    # 4. Load OCR
    ocr = HandwritingOCR("microsoft/trocr-base-handwritten")

    # 5. Recognize cells
    cell_tokens = []
    for row in cell_boxes:
        row_tokens = []
        for cell in row:
            if cell is None:
                row_tokens.append("")
                continue

            (cx1, cy1, w, h) = cell
            cx2 = cx1 + w
            cy2 = cy1 + h

            rgb_crop = matrix_crop[cy1:cy2, cx1:cx2]

            raw_txt = ocr.recognize(rgb_crop)
            cleaned = clean_numeric_prediction(raw_txt)
            final_txt = cleaned

            row_tokens.append(final_txt)
        cell_tokens.append(row_tokens)

    # 6. NEW — collapse extra columns
    cell_tokens = collapse_columns(cell_tokens)

    # 7. Convert to LaTeX
    latex = tokens_to_matrix_latex(cell_tokens)
    return latex


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    latex = recognize_matrix(image_path)
    print("\n===== LaTeX Matrix =====\n")
    print(latex)