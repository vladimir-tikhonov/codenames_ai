import os
import tempfile
import urllib.request
import zipfile
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
from frozendict import frozendict
from keras.layers import Lambda
from keras.models import load_model, Model

from codenames.models.yolov2.post_processing import PostProcessing


def ensure_yolo_models_is_loaded(config: frozendict) -> None:
    model_path = Path(config['modelPath'])
    model_dir = model_path.parent
    model_download_url = config['modelUrl']

    if model_path.is_file():
        print(f'Using an existent yolo model from {model_path}')
        return

    print(f'Downloading yolo model from {model_download_url}...')
    temp_name = os.path.join(tempfile.mkdtemp(), 'model_yolo.zip')
    urllib.request.urlretrieve(model_download_url, temp_name)
    zip_ref = zipfile.ZipFile(temp_name, 'r')
    zip_ref.extractall(model_dir)
    zip_ref.close()
    os.remove(temp_name)
    print('Done.')


class YoloV2:
    def __init__(self, config: frozendict):
        self.config = config

        raw_model = load_model(config['modelPath'], compile=False)
        processed_output = Lambda(PostProcessing(config), name='output_postprocess')(raw_model.output)
        self.model = Model(raw_model.input, processed_output)

    def run(self, image: np.ndarray) -> np.ndarray:
        boxes, _ = self._get_boxes(image.copy())
        return boxes

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (self.config['inputSize'], self.config['inputSize']))
        # yolo normalize
        image = image / 255.
        image = image[:, :, ::-1]
        # cv2 has the channel as bgr, revert to to rgb for Yolo Pass
        image = np.expand_dims(image, 0)

        return image

    def _get_boxes(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        output = self.model.predict(self._preprocess_image(image))[0]

        if output.size == 0:
            return np.array([]), np.array([])

        boxes = output[:, :4]
        scores = output[:, 4]
        return boxes, scores
