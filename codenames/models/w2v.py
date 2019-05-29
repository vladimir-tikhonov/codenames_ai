import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

from frozendict import frozendict
from gensim.models import KeyedVectors


def get_w2v_models(config: frozendict) -> Dict[str, KeyedVectors]:
    ensure_w2v_models_are_loaded(config)

    return {
        'ru': _load_russian_w2v(config)
    }


def ensure_w2v_models_are_loaded(config: frozendict) -> None:
    models_dir = Path(config['modelsDir'])
    rus_model_dir = models_dir / 'rus'
    metadata_file_path = rus_model_dir / 'meta.json'
    model_download_url = config['ruModelUrl']
    model_was_already_downloaded = metadata_file_path.is_file()

    if model_was_already_downloaded:
        print(f'Using an existent russian w2v model from {rus_model_dir}')
        return

    print(f'Downloading russian w2v model from {model_download_url}...')
    temp_name = os.path.join(tempfile.mkdtemp(), 'model.zip')
    urllib.request.urlretrieve(model_download_url, temp_name)
    zip_ref = zipfile.ZipFile(temp_name, 'r')
    zip_ref.extractall(rus_model_dir)
    zip_ref.close()
    os.remove(temp_name)
    print('Done.')


def _load_russian_w2v(config: frozendict) -> KeyedVectors:
    models_dir = Path(config['modelsDir'])
    rus_model_dir = models_dir / 'rus'
    model = KeyedVectors.load_word2vec_format(rus_model_dir / 'model.bin', binary=True)
    model.init_sims(replace=True)
    return model
