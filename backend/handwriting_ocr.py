# handwriting_ocr.py

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import os
import time


class HandwritingOCR:
    """
    Wrapper around microsoft/trocr-base-handwritten.
    Provides a simple .recognize(img) → string interface.
    """

    def __init__(self, model_name="microsoft/trocr-base-handwritten", device=None):
        print(f"[INFO] Loading TrOCR model: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    # ----------------------
    # MAIN OCR FUNCTION
    # ----------------------
    def recognize(self, img):
        """
        Takes BGR or grayscale image crop.
        Returns predicted string.
        """

        # Convert to RGB because TrOCR expects RGB
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -------------------------------------------
        # DEBUG: Save what TrOCR actually sees
        # -------------------------------------------
        debug_path = os.path.expanduser(
            f"~/Desktop/trocr_debug_{int(time.time()*1000)}.png"
        )
        cv2.imwrite(debug_path, rgb)
        print(f"[DEBUG] Saved TrOCR input → {debug_path}")

        pixel_values = (
            self.processor(images=rgb, return_tensors="pt")
            .pixel_values
            .to(self.device)
        )

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return text.strip()

    # ---------------------------------------------------------
    # DIGIT-ONLY RECOGNITION (with token suppression)
    # ---------------------------------------------------------
    def recognize_digit(self, image):

        # Convert to RGB
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------------------------
        # DEBUG: Save digit crop given to TrOCR
        # -------------------------------------------
        debug_path = os.path.expanduser(
            f"~/Desktop/trocr_debug_digit_{int(time.time()*1000)}.png"
        )
        cv2.imwrite(debug_path, rgb)
        print(f"[DEBUG] Saved TrOCR digit input → {debug_path}")

        # Encode image
        pixel_values = (
            self.processor(images=rgb, return_tensors="pt")
            .pixel_values
            .to(self.device)
        )

        # Build digit-only vocabulary
        tokenizer = self.processor.tokenizer
        digit_tokens = [str(i) for i in range(10)]
        digit_ids = tokenizer.convert_tokens_to_ids(digit_tokens)

        # Build suppression list: all vocab except digit IDs
        all_ids = set(range(len(tokenizer)))
        suppress = list(all_ids - set(digit_ids))

        # Run constrained generation
        generated = self.model.generate(
            pixel_values,
            suppress_tokens=suppress,
            max_length = 3
        )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.strip()