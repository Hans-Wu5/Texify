# content_detector.py
import cv2
from pathlib import Path
from .image_preprocessing import preprocess_image

DESKTOP = str(Path.home() / "Desktop")

def save_debug(name, img):
    cv2.imwrite(f"{DESKTOP}/{name}.png", img)


# -----------------------------------------------------------
#  Extract a crop given bounding box
# -----------------------------------------------------------
def extract_block(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


# -----------------------------------------------------------
#  TABLE DETECTOR (unchanged – your accurate version kept)
# -----------------------------------------------------------
def detect_table(gray):
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 50))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel)

    # Find contours
    h_contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(vert,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count long segments
    long_h = [c for c in h_contours if cv2.boundingRect(c)[2] > 80]
    long_v = [c for c in v_contours if cv2.boundingRect(c)[3] > 80]

    # ---- DEBUGGING PRINTS ----
    print(f"Horizontal contours found: {len(h_contours)}")
    print(f"Vertical contours found:   {len(v_contours)}")
    print(f"Long H lines (>80px): {len(long_h)}")
    for i, c in enumerate(long_h):
        x,y,w,h = cv2.boundingRect(c)
        print(f"   H[{i}] = (x={x}, y={y}, w={w}, h={h})")

    print(f"Long V lines (>80px): {len(long_v)}")
    for i, c in enumerate(long_v):
        x,y,w,h = cv2.boundingRect(c)
        print(f"   V[{i}] = (x={x}, y={y}, w={w}, h={h})")

    # ----------------------------------------
    # NEW RULE: Table if H >= 2 AND V >= 2
    # ----------------------------------------
    if len(long_h) < 2 or len(long_v) < 2:
        print("REJECT: Not enough long H/V lines → NOT a table.")
        return []

    print("ACCEPT: Enough horizontal & vertical lines → TABLE DETECTED.")

    # Build combined bounding box
    grid = cv2.add(horiz, vert)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("REJECT: No grid contour found.")
        return []

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    pad_w = int(w * 0.1)
    pad_h = int(h * 0.1)

    H, W = gray.shape
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    return [("table", (x1, y1, x2, y2))]

# -----------------------------------------------------------
#  MATRIX DETECTOR (New – bracket + row grouping)
# -----------------------------------------------------------
def detect_matrix(gray):
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )
    save_debug("debug_matrix_thresh", th)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bracket_candidates = []
    digits = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if h > w * 3 and h > 40:
            bracket_candidates.append((x, y, w, h))
        elif 8 < w < 70 and 10 < h < 70:
            digits.append((x, y, w, h))

    if len(bracket_candidates) < 2 or len(digits) < 3:
        return []

    bracket_candidates = sorted(bracket_candidates, key=lambda b: b[0])
    left_b = bracket_candidates[0]
    right_b = bracket_candidates[-1]

    # --- cluster digits by rows (unchanged) ---
    ys = sorted([d[1] for d in digits])
    rows = []
    row = [ys[0]]
    for y in ys[1:]:
        if abs(y - row[-1]) < 30:
            row.append(y)
        else:
            rows.append(row)
            row = [y]
    rows.append(row)

    if len(rows) < 2:
        return []

    # ---- tighter bounding box logic ----
    digit_x1 = min(d[0] for d in digits)
    digit_x2 = max(d[0] + d[2] for d in digits)

    x1 = min(left_b[0], digit_x1) - 5

    right_bracket_edge = right_b[0] + right_b[2]
    x2 = min(right_bracket_edge + 5, digit_x2 + 20)

    # NEW: use digit bounding boxes instead of row centroids
    digit_y1 = min(d[1] for d in digits)
    digit_y2 = max(d[1] + d[3] for d in digits)

    # tighter vertical padding
    y1 = digit_y1 - 10
    y2 = digit_y2 + 10

    PAD_X = 20   # tweak as needed (5–20 px ideal)
    PAD_Y = 15

    x1 = max(0, x1 - PAD_X)
    y1 = max(0, y1 - PAD_Y)
    x2 = min(gray.shape[1], x2 + PAD_X)
    y2 = min(gray.shape[0], y2 + PAD_Y)

    return [("matrix", (x1, y1, x2, y2), left_b, right_b)]


# -----------------------------------------------------------
#  EQUATION DETECTOR (unchanged)
# -----------------------------------------------------------
def detect_equation(gray):
    # --- 1) Smooth noise ---
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # --- 2) Strong binary threshold (handwriting -> white) ---
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 9
    )

    # --- 3) Close strokes to unify components ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    save_debug("debug_equation_closed", closed)

    # --- 4) Extract external blobs ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # --- 5) Take biggest blob as handwriting region ---
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # --- 6) LARGE padding for safety ---
    pad_w = int(w * 0.6)   # expand 60% horizontally
    pad_h = int(h * 0.8)   # expand 80% vertically

    H, W = gray.shape

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    # Always treat as equation (table/matrix detectors run first)
    return [("equation", (x1, y1, x2, y2))]


# -----------------------------------------------------------
#  MASTER CONTENT DETECTOR
# -----------------------------------------------------------
def detect_content(image_path):
    orig, gray = preprocess_image(image_path)
    save_debug("debug_preprocessed", gray)

    table_boxes = detect_table(gray)
    matrix_boxes = detect_matrix(gray)
    equation_boxes = detect_equation(gray)

    # If multiple types found, pick highest priority:
    # 1. table (strongest)
    # 2. matrix
    # 3. equation
    if table_boxes:
        return orig, table_boxes
    if matrix_boxes:
        return orig, matrix_boxes
    if equation_boxes:
        return orig, equation_boxes

    return orig, []


# -----------------------------------------------------------
#  VISUALIZER
# -----------------------------------------------------------
def visualize_detections(image_path, out="detections_preview.jpg"):
    image, detections = detect_content(image_path)
    vis = image.copy()

    COLORS = {"table": (0, 255, 0), "matrix": (255, 0, 0), "equation": (0, 128, 255)}

    for det in detections:
        label = det[0]

        # MATRIX → ("matrix", box, left_b, right_b)
        if label == "matrix":
            _, (x1, y1, x2, y2), left_b, right_b = det

        # TABLE or EQUATION → ("table", box)
        else:
            _, (x1, y1, x2, y2) = det

        color = COLORS[label]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out_path = str(Path(DESKTOP) / "detections_preview.jpg")
    cv2.imwrite(out_path, vis)
    print("[INFO] Saved preview:", out_path)

    return detections