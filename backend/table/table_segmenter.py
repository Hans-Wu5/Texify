import cv2
import numpy as np
from pathlib import Path

DESKTOP = str(Path.home() / "Desktop")


# -----------------------------------------------------
# Debug visualization
# -----------------------------------------------------
def visualize_cells(gray, cells):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for r, row in enumerate(cells):
        for c, (x, y, w, h) in enumerate(row):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 150, 255), 2)
            cv2.putText(vis, f"{r},{c}", (x + 3, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

    out_path = f"{DESKTOP}/table_cells_debug.png"
    cv2.imwrite(out_path, vis)
    print("[DEBUG] Saved:", out_path)



# -----------------------------------------------------
# Safe collapsing of nearly identical coordinate lines
# -----------------------------------------------------
def collapse(lines, min_gap=10):
    if len(lines) == 0:
        return []

    lines = sorted(lines)
    groups = []
    current = [lines[0]]

    for x in lines[1:]:
        if x - current[-1] <= min_gap:
            current.append(x)
        else:
            groups.append(current)
            current = [x]

    groups.append(current)
    return [int(np.mean(g)) for g in groups]

# -----------------------------------------------------
# Extract clean table cell boxes
# -----------------------------------------------------
def segment_table_cells(gray):

    # 1. Preprocessing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        17, 7
    )

    # --------------------------------------------------------------
    # 2. HORIZONTAL LINES (strong + connected)
    # --------------------------------------------------------------
    h = th.copy()

    # catch thin strokes
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    h1 = cv2.morphologyEx(h, cv2.MORPH_OPEN, h_kernel)

    # connect gaps
    h_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 3))
    h2 = cv2.dilate(h1, h_dilate, iterations=1)

    # unify line
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 5))
    h3 = cv2.morphologyEx(h2, cv2.MORPH_CLOSE, h_close)

    # --------------------------------------------------------------
    # NEW: THICKEN HORIZONTAL LINES BY 20%
    # --------------------------------------------------------------
    thickness = int(gray.shape[0] * 0.01)

    h_thick = cv2.dilate(h3, cv2.getStructuringElement(cv2.MORPH_RECT, (1, thickness)))

    h_lines = h_thick.copy()

    # ---- FILTER OUT SHORT HORIZONTAL LINES ----
    ys, xs = np.where(h_lines > 0)
    if len(ys) > 0:
        unique_y = np.unique(ys)
        lengths = []
        for y in unique_y:
            xs_at_y = xs[ys == y]
            length = xs_at_y.max() - xs_at_y.min()
            lengths.append(length)

        max_len = max(lengths)
        keep_threshold = 0.30 * max_len

        mask = np.zeros_like(h_lines)
        for y in unique_y:
            xs_at_y = xs[ys == y]
            length = xs_at_y.max() - xs_at_y.min()
            if length >= keep_threshold:
                mask[y, xs_at_y] = 255

        h_lines = mask

    '''
    cv2.imwrite(f"{DESKTOP}/debug_h_lines_filtered.png", h_lines)
    print("[DEBUG] Saved filtered horizontal lines")
    '''

    # --------------------------------------------------------------
    # 3. VERTICAL LINES (strong)
    # --------------------------------------------------------------
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel)
    v_lines = v1.copy()

    # ---- FILTER OUT SHORT VERTICAL LINES ----
    ys, xs = np.where(v_lines > 0)
    if len(xs) > 0:
        unique_x = np.unique(xs)
        lengths = []
        for x in unique_x:
            ys_at_x = ys[xs == x]
            length = ys_at_x.max() - ys_at_x.min()
            lengths.append(length)

        max_len_v = max(lengths)
        keep_threshold_v = 0.40 * max_len_v

        mask = np.zeros_like(v_lines)
        for x in unique_x:
            ys_at_x = ys[xs == x]
            length = ys_at_x.max() - ys_at_x.min()
            if length >= keep_threshold_v:
                mask[ys_at_x, x] = 255

        v_lines = mask

    '''
    cv2.imwrite(f"{DESKTOP}/debug_v_lines_filtered.png", v_lines)
    print("[DEBUG] Saved filtered vertical lines")
    '''

    # --------------------------------------------------------------
    # 4. EXTRACT GRID COORDINATES
    # --------------------------------------------------------------
    ys, xs = np.where(h_lines > 0)
    hor_lines = collapse(list(ys))

    ys, xs = np.where(v_lines > 0)
    ver_lines = collapse(list(xs))

    if len(hor_lines) < 2 or len(ver_lines) < 2:
        print("[WARN] Not enough valid grid lines detected.")
        return []

    hor_lines = sorted(hor_lines)
    ver_lines = sorted(ver_lines)

    # --------------------------------------------------------------
    # 5. BUILD CELL BOXES
    # --------------------------------------------------------------
    cells = []
    for i in range(len(hor_lines) - 1):
        row = []
        y1 = hor_lines[i]
        y2 = hor_lines[i + 1]

        for j in range(len(ver_lines) - 1):
            x1 = ver_lines[j]
            x2 = ver_lines[j + 1]
            row.append((x1, y1, x2 - x1, y2 - y1))

        cells.append(row)

    visualize_cells(gray, cells)
    return cells

# -----------------------------------------------------
# Standalone CLI test
# -----------------------------------------------------
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = segment_table_cells(gray)

    for r, row in enumerate(cells):
        print(f"Row {r}: {len(row)} cells")