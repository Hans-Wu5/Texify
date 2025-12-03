# image_preprocessing.py
import cv2
import numpy as np

# ------------------------------------------------------------
# 1. Remove Shadows (Adaptive Background Subtraction)
# ------------------------------------------------------------
def remove_shadows(gray):
    # Estimate background using large morphological opening
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(dilated, 31)

    # Subtract background
    diff = 255 - cv2.absdiff(gray, bg)

    # Normalize
    norm = cv2.normalize(diff, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX)
    return norm


# ------------------------------------------------------------
# 2. Apply CLAHE (Adaptive Local Contrast Boost)
# ------------------------------------------------------------
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ------------------------------------------------------------
# 3. Light Denoising
# ------------------------------------------------------------
def denoise(gray):
    # Bilateral preserves edges (important for handwriting)
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


# ------------------------------------------------------------
# 4. Full Preprocessing Pipeline
# ------------------------------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Steps
    no_shadow = remove_shadows(gray)
    contrasted = apply_clahe(no_shadow)
    clean = denoise(contrasted)

    return img, clean  # return original + processed gray

# ------------------------------------------------------------
# 5. Save for debugging
# ------------------------------------------------------------
def save_preprocessed(image_path, out="preprocessed_debug.jpg"):
    _, proc = preprocess_image(image_path)
    cv2.imwrite(out, proc)
    print("[INFO] Saved preprocessed image to:", out)