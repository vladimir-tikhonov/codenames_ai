import codenames.config.rotation_model as rotation_model_config
import codenames.config.yolo as yolo_config
from codenames.models import ensure_w2v_models_are_loaded, ensure_one_file_model_loaded


def preload_all_models() -> None:
    ensure_w2v_models_are_loaded()
    ensure_one_file_model_loaded('yolo', yolo_config.MODEL_PATH, yolo_config.MODEL_URL)
    ensure_one_file_model_loaded('rotation', rotation_model_config.MODEL_PATH, rotation_model_config.MODEL_URL)


preload_all_models()
