from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Flatten, Dense
from keras.models import Model, load_model as _load_model

import codenames.config.rotation_model as rotation_model_config


def build_model() -> Model:
    base_model = MobileNetV2(
        include_top=False,
        input_shape=(rotation_model_config.IMAGE_SIZE, rotation_model_config.IMAGE_SIZE, 1),
        weights=None
    )
    x = Flatten()(base_model.output)
    final_output = Dense(rotation_model_config.ROTATIONS, activation='softmax', name='rotation_softmax')(x)
    return Model(input=base_model.input, output=final_output)


def load_model() -> Model:
    return _load_model(str(rotation_model_config.MODEL_PATH))
