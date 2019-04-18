from typing import List, Tuple


class Association:
    def __init__(self, association_word: str, associated_words_with_scores: List[Tuple[str, float]]):
        self.association_word = association_word
        self.associated_words = [word for word, _ in associated_words_with_scores]
        self.associated_word_scores = [score for _, score in associated_words_with_scores]

    def add_associated_word(self, word_with_score: Tuple[str, float]):
        self.associated_words.append(word_with_score[0])
        self.associated_word_scores.append(word_with_score[1])

    def has_associated_word(self, word: str) -> bool:
        return word in self.associated_words

    def size(self) -> int:
        return len(self.associated_words)

    def copy(self):
        return Association(self.association_word, list(zip(self.associated_words, self.associated_word_scores)))

    def __repr__(self):
        return f'{self.association_word} -> {self.associated_words}'
