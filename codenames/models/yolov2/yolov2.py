from typing import List

import cv2
import numpy as np
from keras.layers import Lambda
from keras.models import load_model, Model

import codenames.config.yolo as yolo_config
from codenames.models.yolov2.post_processing import PostProcessing


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (yolo_config.INPUT_SIZE, yolo_config.INPUT_SIZE))
    image = image / 255.
    image = image[:, :, ::-1]
    # cv2 has the channel as bgr, revert to to rgb for Yolo Pass
    image = np.expand_dims(image, 0)

    return image


def _is_nonzero_box(box: List[int]) -> bool:
    return box[0] != box[2] and box[1] != box[3]


def _scale_box(image: np.ndarray, box: np.ndarray) -> List[int]:
    image_h, image_w, _ = image.shape
    return [
        max(int(box[0] * image_w), 0),
        max(int(box[1] * image_h), 0),
        int(box[2] * image_w),
        int(box[3] * image_h),
    ]


class YoloV2:
    def __init__(self) -> None:
        raw_model = load_model(str(yolo_config.MODEL_PATH), compile=False)
        processed_output = Lambda(PostProcessing(), name='output_postprocess')(raw_model.output)
        self.model = Model(raw_model.input, processed_output)

    def run(self, image: np.ndarray) -> List[List[int]]:
        output = self.model.predict(_preprocess_image(image))[0]

        if output.size == 0:
            return []

        boxes = list(output[:, :4])
        scaled_boxes = map(lambda box: _scale_box(image, box), boxes)
        return list(filter(_is_nonzero_box, scaled_boxes))
