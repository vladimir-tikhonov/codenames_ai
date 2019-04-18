from .association import Association


def get_guessable_score(association: Association) -> float:
    min_score = min(association.associated_word_scores)
    avg_score = sum(association.associated_word_scores) / len(association.associated_word_scores)
    return 2 * (min_score * avg_score) / (min_score + avg_score)


def get_score(association: Association) -> float:
    return 1 * get_guessable_score(association) + 0.05 * association.size()
