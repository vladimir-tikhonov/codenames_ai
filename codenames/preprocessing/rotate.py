from typing import List, Tuple

import numpy as np

from codenames.models.rotation import rotate_image
from .preprocessor import Preprocessor


class Rotate(Preprocessor):
    def process(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        return [
            (rotate_image(image, 0), '0'),
            (rotate_image(image, 45), '45'),
            (rotate_image(image, 90), '90'),
            (rotate_image(image, 135), '135'),
            (rotate_image(image, 180), '180'),
            (rotate_image(image, 225), '225'),
            (rotate_image(image, 270), '270'),
            (rotate_image(image, 315), '315'),
        ]
