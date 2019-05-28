import os
import tempfile
import urllib.request
import zipfile
from typing import Tuple

import cv2
import numpy as np
from frozendict import frozendict
from keras.layers import Lambda
from keras.models import load_model, Model

from codenames.models.yolov2.post_processing import PostProcessing


def ensure_yolo_models_is_loaded(config: frozendict) -> None:
    model_path = config['modelPath']
    model_dir, _ = os.path.split(model_path)
    model_download_url = config['modelUrl']
    model_was_already_downloaded = os.path.isfile(model_path)

    if model_was_already_downloaded:
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


def draw_boxes(image_in: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
    image = image_in.copy()
    image_h, image_w, _ = image.shape

    color_mod = 255

    for i, box in enumerate(boxes):
        x_min = int(box[0] * image_w)
        y_min = int(box[1] * image_h)
        x_max = int(box[2] * image_w)
        y_max = int(box[3] * image_h)

        if scores is None:
            text = ''
            color_mod = 0
        else:
            text = "(%.1f%%)" % (100 * scores[i])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (color_mod, 255, 0), 2)

        cv2.putText(image,
                    text,
                    (x_min, y_min - 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1e-3 * image_h,
                    (color_mod, 255, 0), 1)
    return image


class YoloV2:
    def __init__(self, config: frozendict):
        self.config = config

        raw_model = load_model(config['modelPath'], compile=False)
        processed_output = Lambda(PostProcessing(config), name='output_postprocess')(raw_model.output)
        self.model = Model(raw_model.input, processed_output)

    def run(self, image_path: str) -> None:
        out_path = os.path.dirname(image_path)
        image = cv2.imread(image_path)

        boxes, scores = self._get_boxes(image.copy())
        image = draw_boxes(image, boxes, scores)
        image_path_without_extension = str(os.path.basename(image_path).split('.')[0])
        out_name = os.path.join(out_path, image_path_without_extension + '_marked.png')
        cv2.imwrite(out_name, image)

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
