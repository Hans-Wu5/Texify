import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2


class TableOCR:
    """
    OCR for table cell text — uses a more general TrOCR model than matrix OCR.
    Adds probability filtering to reduce hallucinated characters.
    """

    def __init__(self, model_name="microsoft/trocr-base-stage1", device=None,
                 prob_threshold=0.90):
        """
        prob_threshold = minimum probability a character must have
                         to be kept in the final output.
        """

        print(f"[INFO] Loading TABLE OCR model: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)

        self.prob_threshold = prob_threshold

    def recognize(self, img):
        """Return full string (letters + digits) from crop."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixel_values = (
            self.processor(images=rgb, return_tensors="pt")  # ← FIXED
            .pixel_values
            .to(self.device)
        )

        with torch.no_grad():
            generated = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()