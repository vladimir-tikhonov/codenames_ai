from typing import List, Tuple

from numpy import ndarray

from codenames.models import YoloV2
from .preprocessor import Preprocessor


def extract_box(image: ndarray, box: ndarray) -> ndarray:
    image_h, image_w, _ = image.shape
    x_min = int(box[0] * image_w)
    y_min = int(box[1] * image_h)
    x_max = int(box[2] * image_w)
    y_max = int(box[3] * image_h)

    return image[y_min:y_max, x_min:x_max]


class ExtractCards(Preprocessor):
    def __init__(self, yolo_model: YoloV2):
        self.yolo_model = yolo_model

    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        boxes = self.yolo_model.run(image)
        return [(extract_box(image, box), str(i)) for i, box in enumerate(boxes)]
