# AI for CodeNames game &middot; [![Build Status](https://travis-ci.org/vladimir-tikhonov/codenames_ai.svg?branch=master)](https://travis-ci.org/vladimir-tikhonov/codenames_ai)

### CLI

```bash
# Basic usage
python -m codenames.cli fox,dog
# Advanced usage
python -m codenames.cli fox,dog --opponent-agents cat,mouse --assassins duck --bystanders cow,goat --lang en
# Show help and exit
python -m codenames.cli --help
```

### API
```bash
# Running in development mode
FLASK_APP=codenames.api FLASK_ENV=development flask run
# Running in production mode
FLASK_APP=codenames.api FLASK_ENV=production flask run

# Example of POST /api/associations request
curl 'http://127.0.0.1:5000/api/associations' -H 'Content-Type: application/json' --data-binary \
     $'{"my_agents": ["fox"], "opponent_agents": ["cat"], "assassins": ["duck"], "bystanders": ["cow"], "lang": "en"}'
```

### Dev cheatsheet
```bash
# Linting
pylint codenames
flake8 codenames
mypy codenames --strict
```
