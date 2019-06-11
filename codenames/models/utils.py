import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List

import numpy as np


def ensure_one_file_model_loaded(model_name: str, model_path: Path, model_url: str) -> None:
    model_dir = model_path.parent

    if model_path.is_file():
        print(f'Using an existing {model_name} model from {model_path}')
        return

    print(f'Downloading {model_name} model from {model_url}...')
    temp_name = os.path.join(tempfile.mkdtemp(), 'model_yolo.zip')
    urllib.request.urlretrieve(model_url, temp_name)
    zip_ref = zipfile.ZipFile(temp_name, 'r')
    zip_ref.extractall(model_dir)
    zip_ref.close()
    os.remove(temp_name)
    print('Done.')


def split_image_into_3_parts(image: np.ndarray) -> List[np.ndarray]:
    height, width, _ = image.shape
    if height > width:
        half_height = height // 2
        quarter_height = half_height // 2
        return [
            image[:half_height].copy(),
            image[quarter_height:-quarter_height].copy(),
            image[half_height:].copy()
        ]
    else:
        half_width = width // 2
        quarter_width = half_width // 2
        return [
            image[:, :half_width].copy(),
            image[:, quarter_width:-quarter_width].copy(),
            image[:, half_width:].copy()
        ]


def extract_box(image: np.ndarray, box: List[int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max]
