from typing import List, Tuple

from numpy import ndarray

from .preprocessor import Preprocessor


class Split(Preprocessor):
    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        height, width, _ = image.shape
        if height > width:
            half_height = height // 2
            quarter_height = half_height // 2
            return [
                (image[:half_height].copy(), 'left'),
                (image[quarter_height:-quarter_height].copy(), 'center'),
                (image[half_height:].copy(), 'right')
            ]
        else:
            half_width = width // 2
            quarter_width = half_width // 2
            return [
                (image[:, :half_width].copy(), 'left'),
                (image[:, quarter_width:-quarter_width].copy(), 'center'),
                (image[:, half_width:].copy(), 'right')
            ]
