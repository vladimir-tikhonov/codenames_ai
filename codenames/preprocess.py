import argparse
import os
import glob
from pathlib import Path
import cv2
from codenames.preprocessing import Split


def preprocess() -> None:
    parser = argparse.ArgumentParser(description='Batch image preprocessing')
    parser.add_argument('--in')
    parser.add_argument('--out')
    args = vars(parser.parse_args())

    input_path = Path(args['in'])
    output_path = Path(args['out'])
    is_dir = os.path.isdir(input_path)

    if not is_dir:
        raise ValueError(f'{input_path} is not an existing directory')
    os.makedirs(output_path, exist_ok=True)

    processor = Split()
    files_to_process = list(map(Path, glob.glob(str(input_path / '*.[jpg][jpeg][png]'))))
    for file_to_process in files_to_process:
        path_to_file, extension = os.path.splitext(str(file_to_process))
        filename = os.path.basename(path_to_file)
        original_image = cv2.imread(str(file_to_process))
        processed_images_with_postfixes = zip(processor.process(original_image), processor.get_image_postfixes())
        for processed_image, postfix in processed_images_with_postfixes:
            processed_image_path = output_path / f'{filename}{"_" + postfix if postfix else ""}{extension}'
            cv2.imwrite(str(processed_image_path), processed_image)


preprocess()
