import os
import tempfile
import urllib.request
import zipfile
from typing import Dict

from gensim.models import KeyedVectors

import codenames.config.w2v as w2v_config


def get_w2v_models() -> Dict[str, KeyedVectors]:
    ensure_w2v_models_are_loaded()

    return {
        'ru': _load_russian_w2v()
    }


def ensure_w2v_models_are_loaded() -> None:
    rus_model_dir = w2v_config.MODELS_DIR / 'rus'
    metadata_file_path = rus_model_dir / 'meta.json'
    model_download_url = w2v_config.RU_MODEL_URL
    model_was_already_downloaded = metadata_file_path.is_file()

    if model_was_already_downloaded:
        print(f'Using an existing russian w2v model from {rus_model_dir}')
        return

    print(f'Downloading russian w2v model from {model_download_url}...')
    temp_name = os.path.join(tempfile.mkdtemp(), 'model.zip')
    urllib.request.urlretrieve(model_download_url, temp_name)
    zip_ref = zipfile.ZipFile(temp_name, 'r')
    zip_ref.extractall(rus_model_dir)
    zip_ref.close()
    os.remove(temp_name)
    print('Done.')


def _load_russian_w2v() -> KeyedVectors:
    rus_model_dir = w2v_config.MODELS_DIR / 'rus'
    model = KeyedVectors.load_word2vec_format(rus_model_dir / 'model.bin', binary=True)
    model.init_sims(replace=True)
    return model
