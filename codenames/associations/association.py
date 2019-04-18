from typing import List, Tuple, Any


class Association:
    association_word: str
    associated_words: List[str]
    associated_word_scores: List[float]
    rival_words: List[str]
    rival_word_scores: List[float]

    def __init__(self, association_word: str, associated_words_with_scores: List[Tuple[str, float]]):
        self.association_word = association_word
        self.associated_words = [word for word, _ in associated_words_with_scores]
        self.associated_word_scores = [score for _, score in associated_words_with_scores]
        self.rival_words = []
        self.rival_word_scores = []

    def add_associated_word(self, word_with_score: Tuple[str, float]) -> None:
        self.associated_words.append(word_with_score[0])
        self.associated_word_scores.append(word_with_score[1])

    def add_rival_word(self, word_with_score: Tuple[str, float]) -> None:
        self.rival_words.append(word_with_score[0])
        self.rival_word_scores.append(word_with_score[1])

    def has_associated_word(self, word: str) -> bool:
        return word in self.associated_words

    def size(self) -> int:
        return len(self.associated_words)

    def copy(self) -> Any:
        return Association(self.association_word, list(zip(self.associated_words, self.associated_word_scores)))

    def __repr__(self) -> str:
        return f'{self.association_word} -> {self.associated_words}'
