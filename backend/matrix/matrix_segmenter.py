# matrix_segmenter.py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# ------------------------------------------------------------
# Helper: check if a contour corresponds to a bracket
# ------------------------------------------------------------
def is_bracket(x, y, w, h, left_b, right_b):
    lx, ly, lw, lh = left_b
    rx, ry, rw, rh = right_b

    # left bracket similarity
    if abs(x - lx) < 25 and abs(h - lh) < 40:
        return True

    # right bracket similarity
    if abs(x - rx) < 25 and abs(h - rh) < 40:
        return True

    return False


# ------------------------------------------------------------
# Segment matrix cells (rows × columns)
# This version:
#   • Ignores brackets entirely
#   • Auto-detects ANY # of rows/columns
#   • Uses Y-clustering for rows
#   • Uses DBSCAN X-clustering for columns
# ------------------------------------------------------------
def segment_matrix_cells(gray, left_b, right_b):
    # --- 1. Threshold for symbol contours ---
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    # --- 2. Filter contours (exclude brackets) ---
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # ignore bracket contours
        if is_bracket(x, y, w, h, left_b, right_b):
            continue

        # keep digit-like shapes
        if 10 < w < 150 and 10 < h < 150:
            digits.append((x, y, w, h))

    if len(digits) == 0:
        return []

    # ------------------------------------------------------------
    # 3. CLUSTER INTO ROWS (by Y-center)
    # ------------------------------------------------------------
    y_centers = np.array([d[1] + d[3] / 2 for d in digits]).reshape(-1, 1)

    row_clusterer = DBSCAN(eps=35, min_samples=1).fit(y_centers)
    row_labels = row_clusterer.labels_

    # group by row label
    rows_dict = {}
    for lbl, d in zip(row_labels, digits):
        rows_dict.setdefault(lbl, []).append(d)

    # sort rows top → bottom
    ordered_rows = [
        rows_dict[k]
        for k in sorted(rows_dict.keys(), key=lambda k: min([x[1] for x in rows_dict[k]]))
    ]

    # ------------------------------------------------------------
    # 4. WITHIN EACH ROW, CLUSTER INTO COLUMNS (by X-center)
    # ------------------------------------------------------------
    final_matrix = []

    for row in ordered_rows:
        x_centers = np.array([d[0] + d[2] / 2 for d in row]).reshape(-1, 1)

        col_clusterer = DBSCAN(eps=40, min_samples=1).fit(x_centers)
        col_labels = col_clusterer.labels_

        cols_dict = {}
        for lbl, d in zip(col_labels, row):
            cols_dict.setdefault(lbl, []).append(d)

        # Sort columns left → right
        ordered_cols = [
            cols_dict[k]
            for k in sorted(cols_dict.keys(), key=lambda k: min([x[0] for x in cols_dict[k]]))
        ]

        # Extract single bounding box per cell
        row_cells = []
        for group in ordered_cols:
            x1 = min([g[0] for g in group])
            y1 = min([g[1] for g in group])
            x2 = max([g[0] + g[2] for g in group])
            y2 = max([g[1] + g[3] for g in group])
            w = x2 - x1
            h = y2 - y1
            row_cells.append((x1, y1, w, h))

        final_matrix.append(row_cells)

    return final_matrix