import argparse
import os
from itertools import chain
from pathlib import Path

import cv2

from codenames.config import read_app_config
from codenames.models import YoloV2
from codenames.preprocessing import Split, ExtractCards, Rotate


def preprocess() -> None:
    config = read_app_config()
    preprocessors = {
        'split': Split(),
        'extract_cards': ExtractCards(YoloV2(config['models']['yolo'])),
        'rotate': Rotate()
    }

    parser = argparse.ArgumentParser(description='Batch image preprocessing')
    parser.add_argument('preprocessor')
    parser.add_argument('--in')
    parser.add_argument('--out')
    args = vars(parser.parse_args())

    input_path = Path(args['in'])
    output_path = Path(args['out'])

    if not input_path.is_dir():
        raise ValueError(f'{input_path} is not an existing directory')
    os.makedirs(output_path, exist_ok=True)

    processor = preprocessors[args['preprocessor']]
    files_to_process = chain(input_path.glob('*.jpg'), input_path.glob('*.jpeg'), input_path.glob('*.png'))
    for file_to_process in files_to_process:
        filename, extension = file_to_process.stem, file_to_process.suffix
        original_image = cv2.imread(str(file_to_process))
        processed_images_with_postfixes = processor.process(original_image)
        for processed_image, postfix in processed_images_with_postfixes:
            processed_image_path = output_path / f'{filename}{"_" + postfix if postfix else ""}{extension}'
            cv2.imwrite(str(processed_image_path), processed_image)


preprocess()
