from typing import Any, Optional, Dict
from jsonschema import validate, ValidationError

ASSOCIATION_REQUEST_SCHEMA = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'properties': {
        'myAgents': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'minItems': 1,
            'maxItems': 10,
            'uniqueItems': True
        },
        'opponentAgents': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'uniqueItems': True
        },
        'assassins': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'uniqueItems': True
        },
        'bystanders': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'uniqueItems': True
        },
        'lang': {
            'type': 'string',
            'enum': ['en', 'ru']
        }
    },
    'required': ['myAgents', 'opponentAgents', 'assassins', 'bystanders', 'lang']
}


def validate_association_request(request: Any) -> Optional[Dict[str, Any]]:
    try:
        validate(request, ASSOCIATION_REQUEST_SCHEMA)
        return None
    except ValidationError as ex:
        return {
            'errorMessage': ex.message,
            'schema': ASSOCIATION_REQUEST_SCHEMA
        }
