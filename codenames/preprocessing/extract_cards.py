from typing import List, Tuple

from numpy import ndarray

from codenames.models import YoloV2
from .preprocessor import Preprocessor


def extract_box(image: ndarray, box: List[int]) -> ndarray:
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max]


class ExtractCards(Preprocessor):
    def __init__(self) -> None:
        self.yolo_model = YoloV2()

    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        boxes = self.yolo_model.run(image)
        return [(extract_box(image, box), str(i)) for i, box in enumerate(boxes)]
