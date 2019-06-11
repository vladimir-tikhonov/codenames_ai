from typing import List

import numpy as np
import pytesseract

from .rotation import rotate_image


def extract_text_from_image(image: np.ndarray) -> List[str]:
    rotations_to_try = [0, 5, -5, 10, -10, 15, -15]
    for rotation in rotations_to_try:
        rotated_image = rotate_image(image, rotation)
        words = pytesseract.image_to_string(rotated_image, lang='rus').split('\n')
        processed_words = [word.strip().lower() for word in words if word]

        if processed_words:
            return processed_words

    return []
