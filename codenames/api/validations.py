from typing import Any, Optional, Dict
from jsonschema import validate, ValidationError

ASSOCIATION_REQUEST_SCHEMA = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'properties': {
        'my_agents': {
            'type': 'array',
            'items': {
                'type': 'string',
            },
            'minItems': 1,
            'uniqueItems': True
        },
        'opponent_agents': {
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
    'required': ['my_agents', 'opponent_agents', 'assassins', 'bystanders', 'lang']
}


def validate_association_request(request: Any) -> Optional[Dict[str, Any]]:
    try:
        validate(request, ASSOCIATION_REQUEST_SCHEMA)
        return None
    except ValidationError as ex:
        return {
            'error_message': ex.message,
            'schema': ASSOCIATION_REQUEST_SCHEMA
        }
