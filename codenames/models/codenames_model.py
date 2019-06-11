from typing import List

import numpy as np

import codenames.config.associations as associations_config
import codenames.config.rotation_model as rotation_model_config
from .ocr import extract_text_from_image
from .rotation import rotate_image, load_model, to_grayscale, scale_image
from .utils import split_image_into_3_parts, extract_box
from .w2v import get_w2v_models
from .yolov2 import YoloV2


class CodenamesModel:
    def __init__(self) -> None:
        self.w2v_models = get_w2v_models()
        self.yolo_model = YoloV2()
        self.rotation_model = load_model()

    def extract_codenames_from_image(self, image: np.ndarray) -> List[str]:
        result: List[str] = []
        image_parts = split_image_into_3_parts(image)
        for image_part in image_parts:
            boxes = self.yolo_model.run(image_part)
            cards_images = [extract_box(image_part, box) for box in boxes]
            for card_image in cards_images:
                processed_card_image = np.expand_dims(
                    scale_image(to_grayscale(card_image), rotation_model_config.IMAGE_SIZE),
                    axis=3
                )
                rotation_step = 360 / rotation_model_config.ROTATIONS
                rotation_index = np.argmax(self.rotation_model.predict(np.array([processed_card_image]))[0])
                target_rotation = int(rotation_step * rotation_index)
                rotated_card_image = rotate_image(card_image, -target_rotation)

                words = extract_text_from_image(rotated_card_image)
                result.extend(filter(lambda word: self._is_valid_word(word, 'ru'), words))

        return list(set(result))

    def _is_valid_word(self, word: str, lang: str) -> bool:
        if len(word) <= 2:
            return False

        w2v_model = self.w2v_models[lang]
        for pos_tag in associations_config.VALID_POS_TAGS_FOR_ASSOCIATED_WORDS:
            word_with_pos_tag = word + '_' + pos_tag
            if word_with_pos_tag in w2v_model.vocab:
                return True

        return False
