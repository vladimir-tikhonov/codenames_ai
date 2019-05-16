import json
from typing import Dict, Any
from frozendict import frozendict


def read_app_config() -> frozendict:
    with open('config.json') as file_handle:
        return _deep_freeze(json.loads(file_handle.read()))


def _deep_freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _deep_freeze_dict(obj)

    if isinstance(obj, list):
        return tuple(obj)

    return obj


def _deep_freeze_dict(dict_to_freeze: Dict[Any, Any]) -> frozendict:
    return frozendict({k: _deep_freeze(v) for k, v in dict_to_freeze.items()})
