# handwriting_ocr.py

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HandwritingOCR:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    # ---------- internal: normalize a small digit/character crop ----------
    def _prepare_image(self, img):
        """
        Accepts:
          - numpy array (gray or BGR)
          - PIL.Image.Image

        Returns a PIL RGB image, 384x384, with the ink centered and thickened.
        """
        # ---- 1. unify to grayscale numpy array ----
        if isinstance(img, np.ndarray):
            if img.ndim == 2:  # already gray
                gray = img.copy()
            else:              # assume BGR from OpenCV
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(img, Image.Image):
            gray = np.array(img.convert("L"))
        else:
            raise TypeError("img must be a numpy array or PIL.Image.Image")

        # small blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # ---- 2. binarize + find ink bounding box ----
        _, th = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        coords = cv2.findNonZero(th)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            th = th[y:y + h, x:x + w]
        # else: keep as-is (very rare / blank cell)

        # ---- 3. slightly thicken strokes ----
        kernel = np.ones((2, 2), np.uint8)
        th = cv2.dilate(th, kernel, iterations=1)

        # invert back to "black ink on white"
        norm = 255 - th

        # ---- 4. resize with padding to 384x384 ----
        target = 384
        h, w = norm.shape
        scale = target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target, target), 255, dtype=np.uint8)
        y0 = (target - new_h) // 2
        x0 = (target - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        pil_img = Image.fromarray(canvas).convert("RGB")
        return pil_img

    # ---------- public: OCR one small crop ----------
    def recognize(self, img) -> str:
        """
        img: numpy crop (gray/BGR) or PIL image of a single symbol/number
        """
        pil_img = self._prepare_image(img)

        pixel_values = self.processor(
            images=pil_img,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=8)

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()