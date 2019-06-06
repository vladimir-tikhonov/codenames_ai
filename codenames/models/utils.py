import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path


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
