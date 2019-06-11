from pathlib import Path
from typing import List

from itertools import chain


def get_all_images_in(directory: Path) -> List[Path]:
    return list(chain(directory.glob('*.jpg'), directory.glob('*.jpeg'), directory.glob('*.png')))
