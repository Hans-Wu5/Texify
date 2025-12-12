# pix2tex_wrapper.py

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

# Try to import pix2tex - delay import check to avoid false warnings
PIX2TEX_AVAILABLE = None
_LatexOCR = None

def _check_pix2tex():
    """Lazy import check for pix2tex"""
    global PIX2TEX_AVAILABLE, _LatexOCR
    if PIX2TEX_AVAILABLE is None:
        try:
            from pix2tex.cli import LatexOCR
            _LatexOCR = LatexOCR
            PIX2TEX_AVAILABLE = True
        except (ImportError, ModuleNotFoundError) as e:
            PIX2TEX_AVAILABLE = False
            import_error_msg = str(e)
            if "pix2tex" in import_error_msg.lower() or "No module named" in import_error_msg:
                print(f"[WARNING] pix2tex not installed. Install with: pip install pix2tex")
    return PIX2TEX_AVAILABLE

# Do initial check
_check_pix2tex()


class EquationOCR:
    """
    Wrapper around pix2tex for equation/formula recognition.
    Supports loading fine-tuned weights.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize pix2tex model.
        
        Args:
            weights_path: Path to fine-tuned weights checkpoint (.pth file)
                         If None, uses default pretrained weights
            device: Device to run on ('cuda', 'cpu', 'mps'). Auto-detects if None
        """
        # Check if pix2tex is available (lazy check)
        if not _check_pix2tex():
            raise ImportError("pix2tex is not installed. Install with: pip install pix2tex")
        
        # Get LatexOCR class
        global _LatexOCR
        if _LatexOCR is None:
            from pix2tex.cli import LatexOCR
            _LatexOCR = LatexOCR
        
        print(f"[INFO] Loading pix2tex model...")
        
        # Choose device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Initialize model
        try:
            if weights_path is None:
                # Use default pretrained model
                print("[INFO] Using default pretrained pix2tex weights")
                self.model = _LatexOCR()
            else:
                # Load fine-tuned weights using pix2tex's native method
                print(f"[INFO] Loading fine-tuned weights from: {weights_path}")
                checkpoint_path = Path(weights_path)
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Weights file not found: {weights_path}")
                
                # pix2tex's LatexOCR expects an 'arguments' parameter (Munch/Namespace)
                # with a 'checkpoint' key. We need to create this properly.
                try:
                    from pix2tex.cli import Munch
                except ImportError:
                    # Fallback if Munch not available - create a simple dict-like class
                    class Munch(dict):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.__dict__ = self
                
                # Create arguments object with checkpoint path
                # Use absolute path to avoid path issues
                checkpoint_abs_path = str(checkpoint_path.resolve())
                
                # Get default config path (pix2tex will use its default if not provided)
                try:
                    from pix2tex.cli import user_data_dir
                    import os
                    data_dir = user_data_dir()
                    default_config = os.path.join(data_dir, 'settings', 'config.yaml')
                    if not os.path.exists(default_config):
                        # Use pix2tex's default config path
                        default_config = 'settings/config.yaml'
                except:
                    default_config = 'settings/config.yaml'
                
                arguments = Munch({
                    'config': default_config,  # Config file path
                    'checkpoint': checkpoint_abs_path,
                    'no_cuda': device == 'cpu',  # Set no_cuda based on device
                    'no_resize': False,  # Allow image resizing
                })
                
                # Initialize model with checkpoint - this is how pix2tex CLI does it
                self.model = _LatexOCR(arguments=arguments)
                print("[INFO] Fine-tuned weights loaded successfully using pix2tex native method")
        except Exception as e:
            # If LatexOCR initialization fails, it might be an import issue
            raise ImportError(f"Failed to initialize pix2tex: {e}. Make sure pix2tex is properly installed: pip install pix2tex")
    
    def recognize(self, image):
        """
        Recognize equation/formula from image and return LaTeX code.
        
        Args:
            image: Can be:
                  - PIL Image
                  - numpy array (BGR or RGB)
                  - OpenCV image (BGR)
                  - Path to image file
        
        Returns:
            str: LaTeX code for the equation
        """
        # Convert input to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert RGBA to RGB if needed
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        
        # Run inference
        try:
            latex_code = self.model(pil_image)
            return latex_code.strip()
        except Exception as e:
            print(f"[ERROR] pix2tex recognition failed: {e}")
            return ""


if __name__ == "__main__":
    # Test the wrapper
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pix2tex_wrapper.py <image_path> [weights_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    weights_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    ocr = EquationOCR(weights_path=weights_path)
    latex = ocr.recognize(image_path)
    print(f"\n===== LaTeX Output =====\n{latex}\n")

