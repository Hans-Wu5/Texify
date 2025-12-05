# matrix_pipeline.py

import cv2
import numpy as np
from Texify.backend.matrix.matrix_handwriting_ocr import MatrixOCR
from Texify.backend.content_detector import detect_content
from .matrix_segmenter import segment_matrix_cells
from .matrix_to_latex import tokens_to_matrix_latex

# ------------------------------
# NUMERIC CLEANUP
# ------------------------------
ALLOWED = set("0123456789.")

def clean_numeric_prediction(txt):
    """Remove non-numeric characters from OCR output."""
    return "".join([c for c in txt if c in ALLOWED]).strip()


def pad_crop(img, pad=6):
    """Expand crop by padding with white space."""
    h, w = img.shape[:2]

    if len(img.shape) == 2:
        canvas = 255 * np.ones((h + pad*2, w + pad*2), dtype=img.dtype)
        canvas[pad:pad+h, pad:pad+w] = img
    else:
        canvas = 255 * np.ones((h + pad*2, w + pad*2, 3), dtype=img.dtype)
        canvas[pad:pad+h, pad:pad+w] = img

    return canvas


def prepare_for_trocr(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    return rgb


def recognize_matrix(image_path):
    # 1. Detect content
    orig, detections = detect_content(image_path)

    if not detections or detections[0][0] != "matrix":
        raise ValueError("No matrix detected")

    (_, (x1, y1, x2, y2), left_b, right_b) = detections[0]

    # 2. Crop matrix
    matrix_crop = orig[y1:y2, x1:x2]
    gray = cv2.cvtColor(matrix_crop, cv2.COLOR_BGR2GRAY)

    # 3. Segment into cells
    cell_boxes = segment_matrix_cells(gray, left_b, right_b)
    if not cell_boxes:
        raise ValueError("Matrix segmentation failed")

    # 4. OCR
    ocr = MatrixOCR("microsoft/trocr-base-handwritten")

    cell_tokens = []
    for row in cell_boxes:
        row_tokens = []
        for (cx1, cy1, w, h) in row:
            cx2 = cx1 + w
            cy2 = cy1 + h

            raw_crop = matrix_crop[cy1:cy2, cx1:cx2]

            padded = pad_crop(raw_crop, pad=20)
            img_ready = prepare_for_trocr(padded)

            raw_txt = ocr.recognize_digit(img_ready)
            cleaned = clean_numeric_prediction(raw_txt)

            row_tokens.append(cleaned)
        cell_tokens.append(row_tokens)

    latex = tokens_to_matrix_latex(cell_tokens)
    return latex


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    latex = recognize_matrix(image_path)
    print("\n===== LaTeX Matrix =====\n")
    print(latex)