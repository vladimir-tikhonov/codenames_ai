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
FLASK_APP=codenames.api:create_app FLASK_ENV=development flask run
# Running in production mode
waitress-serve --call codenames.api:create_app
# Example of POST /api/associations request
curl 'http://127.0.0.1:5000/api/associations' -H 'Content-Type: application/json' --data-binary \
     $'{"myAgents": ["fox"], "opponentAgents": ["cat"], "assassins": ["duck"], "bystanders": ["cow"], "lang": "en"}'
```

### Dev cheatsheet
```bash
# Linting
make lint
# Build docker image for production
make build-docker
# Run production image API on port 8080
make run-api-docker
```
