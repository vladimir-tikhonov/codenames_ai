from typing import List, Tuple

from numpy import ndarray

from codenames.models import split_image_into_3_parts
from .preprocessor import Preprocessor


class Split(Preprocessor):
    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        images = split_image_into_3_parts(image)
        return list(zip(images, ['left', 'center', 'right']))
