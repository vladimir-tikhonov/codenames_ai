from functools import lru_cache
from functools import partial, reduce
from typing import List, Dict, Tuple

from gensim.models import KeyedVectors

import codenames.config.associations as associations_config
from .association import Association
from .metrics import get_score


def build_associations(
        words: List[str],
        rival_words_with_coefficients: List[Tuple[str, float]],
        model: KeyedVectors) -> List[Association]:
    two_word_associations: List[Association] = []
    for i, word_a in enumerate(words):
        for word_b in words[i + 1:]:
            two_word_associations.extend(_get_all_associations_between(word_a, word_b, model))

    all_associations = two_word_associations.copy()
    associations_to_amend = two_word_associations
    while associations_to_amend:
        extended_associations = _extend_associations_with(
            associations_to_amend,
            words,
            model
        )
        all_associations.extend(extended_associations)
        associations_to_amend = extended_associations

    processing_steps = [
        partial(_deduplicate_associations),
        partial(
            _add_rival_words_and_filter_associations,
            rival_words_with_coefficients=rival_words_with_coefficients,
            model=model
        ),
        partial(_remove_invalid_associations),
        partial(_remove_similar_explanation_of_the_same_thing),
    ]

    return reduce(
        lambda associations, processing_step: processing_step(associations),  # type: ignore
        processing_steps,
        all_associations
    )


@lru_cache(maxsize=1024)
def _get_all_associations_between(
        word_a: str,
        word_b: str,
        model: KeyedVectors) -> List[Association]:
    similar_words_with_scores_a = _get_similar_words_with_scores(word_a, model)
    similar_words_with_scores_b = _get_similar_words_with_scores(word_b, model)
    association_words = set(similar_words_with_scores_a.keys()).intersection(similar_words_with_scores_b.keys())

    result = []
    for association_word in association_words:
        new_association = Association(association_word, [
            (word_a, similar_words_with_scores_a[association_word]),
            (word_b, similar_words_with_scores_b[association_word])
        ])
        result.append(new_association)

    return result


def _extend_associations_with(
        associations: List[Association],
        words: List[str],
        model: KeyedVectors) -> List[Association]:
    valid_pos_tags_for_associations = associations_config.VALID_POS_TAGS_FOR_ASSOCIATIONS
    valid_pos_tags_for_associated_words = associations_config.VALID_POS_TAGS_FOR_ASSOCIATED_WORDS

    result = []
    for association in associations:
        association_word_with_tag = _add_pos_tag(
            association.association_word,
            _get_most_likely_pos_tag(association.association_word, valid_pos_tags_for_associations, model)
        )
        for word in words:
            if association.has_associated_word(word):
                continue

            word_with_tag = _add_pos_tag(
                word,
                _get_most_likely_pos_tag(word, valid_pos_tags_for_associated_words, model)
            )
            similarity = model.similarity(association_word_with_tag, word_with_tag)
            if similarity < associations_config.MIN_SIMILARITY_SCORE:
                continue

            new_association = association.copy()
            new_association.add_associated_word((word, similarity))
            result.append(new_association)

    return result


def _remove_invalid_associations(associations: List[Association]) -> List[Association]:
    return list(filter(is_valid_association, associations))


def is_valid_association(association: Association) -> bool:
    association_matches_associated_words = any(
        [associated_word in association.association_word or association.association_word in associated_word
         for associated_word in association.associated_words]
    )
    if association_matches_associated_words:
        return False

    association_contains_invalid_characters = '-' in association.association_word
    if association_contains_invalid_characters:
        return False

    return True


def _deduplicate_associations(associations: List[Association]) -> List[Association]:
    result: List[Association] = []
    associations_sorted_by_size = sorted(associations, key=lambda a: a.size(), reverse=True)

    for association in associations_sorted_by_size:
        is_useless = any(
            [_is_duplicate_or_subset(association, existing_association) for existing_association in result]
        )
        if not is_useless:
            result.append(association)

    return result


def _is_duplicate_or_subset(smaller_association: Association, bigger_association: Association) -> bool:
    if smaller_association.association_word != bigger_association.association_word:
        return False

    smaller_words_set = set(smaller_association.associated_words)
    bigger_words_set = set(bigger_association.associated_words)
    return smaller_words_set == bigger_words_set or smaller_words_set.issubset(bigger_words_set)


def _remove_similar_explanation_of_the_same_thing(associations: List[Association]) -> List[Association]:
    associations_grouped_by_associated_words: Dict[str, List[Association]] = {}
    for association in associations:
        groping_key = ','.join(association.associated_words)
        if groping_key in associations_grouped_by_associated_words:
            associations_grouped_by_associated_words[groping_key].append(association)
        else:
            associations_grouped_by_associated_words[groping_key] = [association]

    result: List[Association] = []
    for _, grouped_associations in associations_grouped_by_associated_words.items():
        associations_to_keep = associations_config.MAX_DIFFERENT_EXPLANATIONS_OF_THE_SAME_THING
        if len(grouped_associations) <= associations_to_keep:
            result.extend(grouped_associations)
        else:
            sorted_associations = sorted(grouped_associations, key=get_score, reverse=True)
            result.extend(sorted_associations[:associations_to_keep])
    return result


def _add_rival_words_and_filter_associations(
        associations: List[Association],
        rival_words_with_coefficients: List[Tuple[str, float]],
        model: KeyedVectors) -> List[Association]:
    result: List[Association] = []
    valid_pos_tags_for_associations = associations_config.VALID_POS_TAGS_FOR_ASSOCIATIONS
    valid_pos_tags_for_associated_words = associations_config.VALID_POS_TAGS_FOR_ASSOCIATED_WORDS

    for association in associations:
        min_associated_word_score = min(association.associated_word_scores)
        association_word_with_tag = _add_pos_tag(
            association.association_word,
            _get_most_likely_pos_tag(association.association_word, valid_pos_tags_for_associations, model)
        )

        association_is_too_dangerous = False
        for rival_word, rival_word_coefficient in rival_words_with_coefficients:
            rival_word_with_pos_tag = _add_pos_tag(
                rival_word,
                _get_most_likely_pos_tag(rival_word, valid_pos_tags_for_associated_words, model)
            )

            rival_word_score = model.similarity(association_word_with_tag, rival_word_with_pos_tag)
            scaled_rival_word_score = rival_word_coefficient * rival_word_score
            if scaled_rival_word_score > min_associated_word_score:
                association_is_too_dangerous = True
                break

            association.add_rival_word((rival_word, float(rival_word_score)))

        if not association_is_too_dangerous:
            result.append(association)

    return result


def _get_similar_words_with_scores(word: str, model: KeyedVectors) -> Dict[str, float]:
    scores_by_similar_word: Dict[str, float] = {}
    valid_pos_tags_for_associations = set(associations_config.VALID_POS_TAGS_FOR_ASSOCIATIONS)
    valid_pos_tags_for_associated_words = associations_config.VALID_POS_TAGS_FOR_ASSOCIATED_WORDS

    most_likely_pos_tag = _get_most_likely_pos_tag(word, valid_pos_tags_for_associated_words, model)
    word_with_pos_tag = _add_pos_tag(word, most_likely_pos_tag)
    similar_words_with_scores_and_tags = model.similar_by_word(
        word_with_pos_tag,
        topn=associations_config.SIMILAR_WORDS_TO_CONSIDER,
        restrict_vocab=associations_config.AMOUNT_OF_TOP_WORDS_FOR_ASSOCIATIONS
    )
    similar_words_with_scores = [(_remove_pos_tag(similar_word), score) for similar_word, score
                                 in similar_words_with_scores_and_tags
                                 if _get_pos_tag(similar_word) in valid_pos_tags_for_associations and
                                 similar_word != word and
                                 score > associations_config.MIN_SIMILARITY_SCORE]

    for similar_word, score in similar_words_with_scores:
        if similar_word in scores_by_similar_word:
            scores_by_similar_word[similar_word] = max(scores_by_similar_word[similar_word], score)
        else:
            scores_by_similar_word[similar_word] = score

    return scores_by_similar_word


def _add_pos_tag(word: str, tag: str) -> str:
    return f'{word}_{tag}'


def _remove_pos_tag(word: str) -> str:
    return word.split('_')[0]


def _get_pos_tag(word: str) -> str:
    return word.split('_')[1]


def _get_most_likely_pos_tag(word: str, pos_tags: List[str], model: KeyedVectors) -> str:
    post_tags_with_frequencies: List[Tuple[str, float]] = []
    for pos_tag in pos_tags:
        word_with_pos_tag = _add_pos_tag(word, pos_tag)
        if word_with_pos_tag not in model.vocab:
            continue

        post_tags_with_frequencies.append((pos_tag, model.vocab[word_with_pos_tag].count))

    if not post_tags_with_frequencies:
        raise ValueError(f'Word {word} is not in the vocabulary')

    return max(post_tags_with_frequencies, key=lambda tag_with_frequency: tag_with_frequency[1])[0]
