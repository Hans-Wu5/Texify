# equation_pipeline.py

import cv2
import numpy as np
from pathlib import Path
from .pix2tex_wrapper import EquationOCR
from ..content_detector import detect_content

DESKTOP = str(Path.home() / "Desktop")

# ============================================================
# Contour-based cropper (UNCHANGED from your version)
# ============================================================
def detect_equation_contour(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 7
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    xs, ys, xe, ye = [], [], [], []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w * h < 80:
            continue
        if w > gray.shape[1] * 0.9 and h > gray.shape[0] * 0.9:
            continue

        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)

    if not xs:
        return None

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xe)
    y2 = max(ye)

    pad_w = int((x2 - x1) * 0.20)
    pad_h = int((y2 - y1) * 0.20)

    H, W = gray.shape
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W, x2 + pad_w)
    y2 = min(H, y2 + pad_h)

    if (x2 - x1) < 30 or (y2 - y1) < 30:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# ORIGINAL helper (UNCHANGED)
# ============================================================
def prepare_equation_image(crop):
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb


# ============================================================
# Weight-finder (UNCHANGED)
# ============================================================
def _find_weights_file(weights_path=None):
    if weights_path:
        wp = Path(weights_path)
        if wp.exists():
            return str(wp)

        resolved = Path(__file__).parent.parent / weights_path
        if resolved.exists():
            return str(resolved)

        return weights_path

    local_dir = Path(__file__).parent
    local_weights = list(local_dir.glob("*.pth")) + list(local_dir.glob("*.pt"))
    if local_weights:
        return str(local_weights[0])

    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        weight_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
        if weight_files:
            preferred = [f for f in weight_files if "mixed" in f.name.lower() or "finetuned" in f.name.lower()]
            if preferred:
                return str(preferred[0])
            return str(weight_files[0])

    return None


# ============================================================
# MAIN FUNCTION (patched minimally, crop + beautify added)
# ============================================================
def recognize_equation(image_path, weights_path=None, force_equation=False):
    from PIL import Image

    detected_weights = _find_weights_file(weights_path)
    if detected_weights:
        print(f"[INFO] Using weights: {detected_weights}")

    orig, detections = detect_content(image_path)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if len(orig.shape) == 3 else orig

    contour_box = detect_equation_contour(gray)

    if contour_box:
        print("[INFO] Using contour-based equation crop")
        x1, y1, x2, y2 = contour_box
        eq_crop = orig[y1:y2, x1:x2]
        '''
        debug_path = f"{DESKTOP}/equation_contour_crop.png"
        cv2.imwrite(debug_path, eq_crop)
        print(f"[DEBUG] Saved contour crop → {debug_path}")
        '''
        pil_image = Image.fromarray(eq_crop)

    else:
        print("[WARNING] Contour detection failed → using full image")

        if len(orig.shape) == 3:
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(orig_rgb)
        else:
            pil_image = Image.fromarray(orig).convert('RGB')

        debug_path = f"{DESKTOP}/equation_crop.png"
        pil_image.save(debug_path)
        print(f"[DEBUG] Saved fallback crop → {debug_path}")

    # Save the exact image fed to pix2tex
    debug_pix2tex_path = f"{DESKTOP}/equation_pix2tex_input.png"
    pil_image.save(debug_pix2tex_path)
    print(f"[DEBUG] Saved final pix2tex input → {debug_pix2tex_path}")

    ocr = EquationOCR(weights_path=detected_weights)
    latex_code = ocr.recognize(pil_image)

    if latex_code and not latex_code.startswith("$"):
        latex_code = f"${latex_code}$"

    return latex_code


# ============================================================
# CLI (UNCHANGED)
# ============================================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Process equation images and generate LaTeX")
    parser.add_argument("image_path", help="Path to the equation image")
    parser.add_argument("--weights", "-w", help="Path to fine-tuned weights file (auto-detected if not provided)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force equation processing even if content detector classifies as table/matrix")

    args = parser.parse_args()

    try:
        latex = recognize_equation(
            args.image_path,
            weights_path=args.weights,
            force_equation=args.force
        )
        print("\n===== LaTeX Equation =====\n")
        print(latex)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()