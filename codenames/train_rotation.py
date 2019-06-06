import argparse
import random
from itertools import chain
from pathlib import Path
from typing import List, Generator

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import codenames.config.rotation_model as rotation_model_config
from codenames.models.rotation import scale_image, rotate_image, build_model, to_grayscale


def image_generator(images: List[np.ndarray]) -> Generator[np.ndarray, None, None]:
    while True:
        rotation_step = 360 / rotation_model_config.ROTATIONS

        current_images_batch = []
        current_labels_batch = []

        for image in images:
            for current_rotation_id in range(rotation_model_config.ROTATIONS):
                current_rotation = int(rotation_step * current_rotation_id)
                processed_image = scale_image(
                    rotate_image(to_grayscale(image), current_rotation),
                    rotation_model_config.IMAGE_SIZE
                )

                current_images_batch.append(
                    processed_image.reshape((processed_image.shape[0], processed_image.shape[1], 1))
                )
                current_labels_batch.append(
                    to_categorical(current_rotation_id, num_classes=rotation_model_config.ROTATIONS)
                )

                if len(current_images_batch) >= rotation_model_config.TRAIN_BATCH_SIZE:
                    yield (np.array(current_images_batch), np.array(current_labels_batch))
                    current_images_batch = []
                    current_labels_batch = []


def train() -> None:
    parser = argparse.ArgumentParser(description='Train rotation model')
    parser.add_argument('--in')
    parser.add_argument('--out')
    args = vars(parser.parse_args())

    input_path = Path(args['in'])
    output_path = Path(args['out'])

    if not input_path.is_dir():
        raise ValueError(f'{input_path} is not an existing directory')

    files_to_process = chain(input_path.glob('*.jpg'), input_path.glob('*.jpeg'), input_path.glob('*.png'))
    images = [cv2.imread(str(image_path)) for image_path in files_to_process]

    random.seed(42)
    random.shuffle(images)
    train_images, test_images = train_test_split(images, random_state=42, test_size=0.2)
    model = build_model()
    optimizer = Adam(0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    stop_early = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)
    save_on_epoch_end = ModelCheckpoint(str(output_path) + '.tmp', save_best_only=True, verbose=1)

    model.fit_generator(
        generator=image_generator(train_images),
        steps_per_epoch=(len(train_images) * rotation_model_config.ROTATIONS) // rotation_model_config.TRAIN_BATCH_SIZE,
        epochs=rotation_model_config.TRAIN_EPOCH,
        verbose=1,
        validation_data=image_generator(test_images),
        validation_steps=(len(test_images) * rotation_model_config.ROTATIONS) // rotation_model_config.TRAIN_BATCH_SIZE,
        callbacks=[stop_early, save_on_epoch_end]
    )

    model.save(str(output_path))


train()
