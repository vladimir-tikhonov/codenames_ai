from codenames.config import read_app_config
from codenames.models import ensure_w2v_models_are_loaded, ensure_yolo_models_is_loaded


def preload_all_models() -> None:
    app_config = read_app_config()
    models_config = app_config['models']
    ensure_w2v_models_are_loaded(models_config['w2v'])
    ensure_yolo_models_is_loaded(models_config['yolo'])


preload_all_models()
