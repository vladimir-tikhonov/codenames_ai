from typing import List, TypeVar, Tuple

from numpy import ndarray

T = TypeVar('T')


def flatten(nested_list: List[List[T]]) -> List[T]:
    return [item for sublist in nested_list for item in sublist]


class Preprocessor:
    def process(self, image: ndarray) -> List[Tuple[ndarray, str]]:
        raise NotImplementedError()

    def process_batch(self, images: List[ndarray]) -> List[Tuple[ndarray, str]]:
        return flatten(list(map(self.process, images)))
