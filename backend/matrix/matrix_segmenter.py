# matrix_segmenter.py
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path

DESKTOP = str(Path.home() / "Desktop")

def is_bracket(x, y, w, h, left_b, right_b):
    lx, ly, lw, lh = left_b
    rx, ry, rw, rh = right_b

    if abs(x - lx) < 25 and abs(h - lh) < 40:
        return True
    if abs(x - rx) < 25 and abs(h - rh) < 40:
        return True
    return False


def visualize_cells(gray, final_matrix):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for r, row in enumerate(final_matrix):
        for c, (x, y, w, h) in enumerate(row):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{r},{c}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out_path = f"{DESKTOP}/matrix_cells_debug.png"
    cv2.imwrite(out_path, vis)
    print(f"[DEBUG] Saved matrix visualization → {out_path}")


def segment_matrix_cells(gray, left_b, right_b):
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if is_bracket(x, y, w, h, left_b, right_b):
            continue

        if w > 2.5 * h:   # horizontal noise
            continue

        if h < 12 or w < 6:  # tiny noise
            continue

        if 3 < w < 150 and 10 < h < 150:
            digits.append((x, y, w, h))

    if len(digits) == 0:
        return []

    # -------- ROW CLUSTER ----------
    y_centers = np.array([d[1] + d[3] / 2 for d in digits]).reshape(-1, 1)
    row_clusterer = DBSCAN(eps=35, min_samples=1).fit(y_centers)
    row_labels = row_clusterer.labels_

    rows_dict = {}
    for lbl, d in zip(row_labels, digits):
        rows_dict.setdefault(lbl, []).append(d)

    ordered_rows = [
        rows_dict[k]
        for k in sorted(rows_dict.keys(), key=lambda k: min([x[1] for x in rows_dict[k]]))
    ]

    # -------- COL CLUSTER ----------
    final_matrix = []

    for row in ordered_rows:
        x_centers = np.array([d[0] + d[2] / 2 for d in row]).reshape(-1, 1)
        col_clusterer = DBSCAN(eps=50, min_samples=1).fit(x_centers)
        col_labels = col_clusterer.labels_

        cols_dict = {}
        for lbl, d in zip(col_labels, row):
            cols_dict.setdefault(lbl, []).append(d)

        ordered_cols = [
            cols_dict[k]
            for k in sorted(cols_dict.keys(), key=lambda k: min([x[0] for x in cols_dict[k]]))
        ]

        row_cells = []
        for group in ordered_cols:
            x1 = min([g[0] for g in group])
            y1 = min([g[1] for g in group])
            x2 = max([g[0] + g[2] for g in group])
            y2 = max([g[1] + g[3] for g in group])
            row_cells.append((x1, y1, x2 - x1, y2 - y1))

        final_matrix.append(row_cells)

    # ============================================================
    # PART 5 — GLOBAL FILTER
    # ============================================================

    # Step 1 — flatten all cells globally
    all_cells = [cell for row in final_matrix for cell in row]

    if not all_cells:
        return final_matrix

    # Step 2 — find largest GLOBAL area
    global_largest = max([w*h for (_,_,w,h) in all_cells])

    # Step 3 — compute global threshold
    threshold = 0.10 * global_largest  # keep only ≥10% of largest

    # Step 4 — filter per row using GLOBAL threshold
    cleaned = []
    for row in final_matrix:
        filtered_row = [(x, y, w, h) for (x, y, w, h) in row if (w * h) >= threshold]

        # <<< NEW: skip empty rows entirely
        if len(filtered_row) == 0:
            continue

        cleaned.append(filtered_row)

    final_matrix = cleaned

    # ============================================================

    visualize_cells(gray, final_matrix)
    return final_matrix