import os
import urllib.request
import tempfile
import zipfile
from configparser import SectionProxy
from typing import Dict
from gensim.models import KeyedVectors


def get_w2v_models(models_config: SectionProxy) -> Dict[str, KeyedVectors]:
    return {
        'ru': load_russian_w2v(models_config)
    }


def load_russian_w2v(models_config: SectionProxy) -> KeyedVectors:
    rus_model_dir = os.path.join(models_config.get('W2VDir'), 'rus')
    metadata_file_path = os.path.join(rus_model_dir, 'meta.json')
    model_download_url = models_config.get('RusW2VModelUrl')
    model_was_already_downloaded = os.path.isfile(metadata_file_path)
    if model_was_already_downloaded:
        print(f'Using an existent russian w2v model from {rus_model_dir}')
    else:
        print(f'Downloading russian w2v model from {model_download_url}...')
        temp_name = os.path.join(tempfile.mkdtemp(), 'model.zip')
        urllib.request.urlretrieve(model_download_url, temp_name)
        zip_ref = zipfile.ZipFile(temp_name, 'r')
        zip_ref.extractall(rus_model_dir)
        zip_ref.close()
        os.remove(temp_name)
        print('Done')
    model = KeyedVectors.load_word2vec_format(os.path.join(rus_model_dir, 'model.bin'), binary=True)
    model.init_sims(replace=True)
    return model
