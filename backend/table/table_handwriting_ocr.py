# table_ocr.py

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2


class TableOCR:
    """
    General OCR for table text.
    Unlike matrix OCR, this keeps the FULL vocabulary.
    Ideal for words, labels, headers, mixed content.
    """

    def __init__(self, model_name="microsoft/trocr-base-handwritten", device=None):
        print(f"[INFO] Loading Table OCR model: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)

    def recognize(self, img):
        """
        BGR/GRAY → text string.
        No digit restriction.
        """

        # BGR/gray → RGB
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        pixel_values = (
            self.processor(images=rgb, return_tensors="pt")
            .pixel_values.to(self.device)
        )

        # Generate text
        with torch.no_grad():
            ids = self.model.generate(pixel_values)
            txt = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

        return txt.strip()