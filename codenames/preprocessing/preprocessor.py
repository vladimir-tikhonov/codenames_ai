from typing import List, TypeVar

from numpy import ndarray

T = TypeVar('T')


def flatten(nested_list: List[List[T]]) -> List[T]:
    return [item for sublist in nested_list for item in sublist]


class Preprocessor:
    @staticmethod
    def get_image_postfixes() -> List[str]:
        return ['']

    def process(self, image: ndarray) -> List[ndarray]:
        raise NotImplementedError()

    def process_batch(self, images: List[ndarray]) -> List[ndarray]:
        return flatten(list(map(self.process, images)))
