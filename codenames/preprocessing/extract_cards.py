from typing import List, Tuple

from numpy import ndarray

from codenames.models import YoloV2, extract_box
from .preprocessor import Preprocessor


class ExtractCards(Preprocessor):
    def __init__(self) -> None:
        self.yolo_model = YoloV2()

    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        boxes = self.yolo_model.run(image)
        return [(extract_box(image, box), str(i)) for i, box in enumerate(boxes)]
