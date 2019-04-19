from .association import Association


def get_guessable_score(association: Association) -> float:
    min_score = min(association.associated_word_scores)
    avg_score = sum(association.associated_word_scores) / len(association.associated_word_scores)
    return 2 * (min_score * avg_score) / (min_score + avg_score)


def get_confusion_score(association: Association) -> float:
    if not association.rival_word_scores:
        return 0

    max_rival_score = max(association.rival_word_scores)
    min_associated_word_score = min(association.associated_word_scores)
    return max_rival_score * (max_rival_score / min_associated_word_score)


def get_score(association: Association) -> float:
    return 1 * get_guessable_score(association) + \
           0.05 * association.size() - \
           0.2 * get_confusion_score(association)
