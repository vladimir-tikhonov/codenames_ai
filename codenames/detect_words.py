import argparse
from pathlib import Path

import cv2

from codenames.models import CodenamesModel
from codenames.preprocessing import get_all_images_in


def detect_words() -> None:
    model = CodenamesModel()

    parser = argparse.ArgumentParser(description='Detect words on image')
    parser.add_argument('--in')
    args = vars(parser.parse_args())
    input_path = Path(args['in'])

    files_to_process = get_all_images_in(input_path)
    for file_to_process in files_to_process:
        image = cv2.imread(str(file_to_process))
        words = model.extract_codenames_from_image(image)
        print(f'{file_to_process} ({len(words)} words):')
        for word in words:
            print(word)


detect_words()
