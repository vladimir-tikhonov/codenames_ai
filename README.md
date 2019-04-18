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

### Dev cheatsheet
```bash
# Linting
pylint codenames
flake8 codenames
mypy codenames --strict
```
