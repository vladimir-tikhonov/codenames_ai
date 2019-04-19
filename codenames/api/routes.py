from typing import Dict, Any
from flask import Flask, jsonify, request
from codenames.config import read_app_config
from codenames.models import get_w2v_models
from codenames.associations import build_associations, prepare_rival_words_with_coefficients, \
    Association, get_score, get_guessable_score, get_confusion_score
from .validations import validate_association_request


def init_routes(app: Flask) -> None:
    app_config = read_app_config()
    models_config = app_config['models']
    associations_config = app_config['associations']

    w2v_models = get_w2v_models(models_config)

    @app.route('/api/associations', methods=['POST'])
    def generate_associations() -> Any:  # pylint: disable=unused-variable
        validation_error = validate_association_request(request.json)
        if validation_error:
            response = jsonify(validation_error)
            response.status_code = 400
            return response

        my_agents = request.json['my_agents']
        assassins = request.json['assassins']
        opponent_agents = request.json['opponent_agents']
        bystanders = request.json['bystanders']
        lang = request.json['lang']

        rival_words_with_coefficients = prepare_rival_words_with_coefficients(
            assassins, opponent_agents, bystanders, associations_config
        )

        associations = build_associations(
            my_agents,
            rival_words_with_coefficients,
            w2v_models[lang],
            associations_config
        )
        print(list(map(_serialize_association, associations)))
        return jsonify({
            'associations': list(map(_serialize_association, associations))
        })


def _serialize_association(association: Association) -> Dict[str, Any]:
    return {
        'association_word': association.association_word,
        'associated_words': association.associated_words,
        'rival_words': association.rival_words,
        'rival_word_scores': association.rival_word_scores,
        'overall_score': get_score(association),
        'guessable_score': get_guessable_score(association),
        'confusion_score': get_confusion_score(association)
    }
