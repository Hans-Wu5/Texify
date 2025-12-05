def prepare_for_trocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # light denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # restore 3 channels
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb