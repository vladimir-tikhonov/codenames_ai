from typing import Any, List, Dict

import cv2
import numpy as np
from flask import Flask, jsonify, request

from codenames.associations import build_associations, prepare_rival_words_with_coefficients, \
    Association, get_score, get_guessable_score, get_confusion_score
from codenames.models import CodenamesModel
from .validations import validate_association_request


def init_routes(app: Flask) -> None:
    codenames_model = CodenamesModel()

    @app.route('/api/associations', methods=['POST'])
    def generate_associations() -> Any:  # pylint: disable=unused-variable
        validation_error = validate_association_request(request.json)
        if validation_error:
            response = jsonify(validation_error)
            response.status_code = 400
            return response

        my_agents = preprocess_words(request.json['myAgents'])
        assassins = preprocess_words(request.json['assassins'])
        opponent_agents = preprocess_words(request.json['opponentAgents'])
        bystanders = preprocess_words(request.json['bystanders'])
        lang = request.json['lang']

        rival_words_with_coefficients = prepare_rival_words_with_coefficients(
            assassins, opponent_agents, bystanders
        )

        try:
            associations = build_associations(
                my_agents,
                rival_words_with_coefficients,
                codenames_model.w2v_models[lang]
            )
            return jsonify({
                'associations': list(map(_serialize_association, associations))
            })
        except Exception as ex:  # pylint: disable=broad-except
            response = jsonify({'errorMessage': str(ex)})
            response.status_code = 500
            return response

    @app.route('/api/detect', methods=['POST'])
    def detect_words_on_image() -> Any:  # pylint: disable=unused-variable
        if 'image' not in request.files:
            response = jsonify({'errorMessage': 'Image is missing'})
            response.status_code = 400
            return response
        image_file = request.files['image']
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        words = codenames_model.extract_codenames_from_image(image)
        return jsonify({'words': words})


def _serialize_association(association: Association) -> Dict[str, Any]:
    return {
        'associationWord': association.association_word,
        'associatedWords': association.associated_words,
        'rivalWords': association.rival_words,
        'rivalWordScores': association.rival_word_scores,
        'overallScore': get_score(association),
        'guessableScore': get_guessable_score(association),
        'confusionScore': get_confusion_score(association)
    }


def preprocess_words(words: List[str]) -> List[str]:
    return list(map(preprocess_word, words))


def preprocess_word(word: str) -> str:
    return word.lower().strip()
