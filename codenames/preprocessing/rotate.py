from typing import List, Tuple

import numpy as np

import codenames.config.rotation_model as rotation_model_config
from codenames.models.rotation import rotate_image, load_model, to_grayscale, scale_image
from .preprocessor import Preprocessor


class Rotate(Preprocessor):
    def __init__(self) -> None:
        self.model = load_model()

    def process(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        processed_image = np.expand_dims(
            scale_image(to_grayscale(image), rotation_model_config.IMAGE_SIZE),
            axis=3
        )
        rotation_step = 360 / rotation_model_config.ROTATIONS
        rotation_index = np.argmax(self.model.predict(np.array([processed_image]))[0])
        target_rotation = int(rotation_step * rotation_index)
        return [(rotate_image(image, -target_rotation), str(target_rotation))]
