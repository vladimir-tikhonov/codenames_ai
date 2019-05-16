from typing import List
from numpy import ndarray
from .preprocessor import Preprocessor


class Split(Preprocessor):
    @staticmethod
    def get_image_postfixes() -> List[str]:
        return ['left', 'center', 'right']

    def process(self, image: ndarray) -> List[ndarray]:
        height, width, _ = image.shape
        if height > width:
            half_height = height // 2
            quarter_height = half_height // 2
            return [
                image[:half_height].copy(),
                image[quarter_height:-quarter_height].copy(),
                image[half_height:].copy()
            ]
        else:
            half_width = width // 2
            quarter_width = half_width // 2
            return [
                image[:, :half_width].copy(),
                image[:, quarter_width:-quarter_width].copy(),
                image[:, half_width:].copy()
            ]
