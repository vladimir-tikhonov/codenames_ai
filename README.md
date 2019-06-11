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
# Example of POST /api/detect request
curl 'http://127.0.0.1:5000/api/detect' -F 'image=@/path/to/the/image/photo.jpg'
```

### Models
- `word2vec` in order to find similar words. Pre-trained russian version was taken from [here](https://rusvectores.org/ru/models/).
- `YoloV2` in order to locate cards in image. Model was trained using an implementation from [ksanjeevan/dourflow](https://github.com/ksanjeevan/dourflow). 
Trained model is available [here](https://www.dropbox.com/s/obn7woabs8wav11/yolo_v2.h5.zip?raw=1).
Data for training is available [here](https://www.dropbox.com/s/tzwvdonv3c4mnpa/train_yolo.zip?raw=1).
- `rotation` model (based on `MobileNetV2` architecture) in order to detect cards rotation.
Trained model is available [here](https://www.dropbox.com/s/qwacqxqaaohpmze/rotation_model.zip?raw=1).
Data for training is available [here](https://www.dropbox.com/s/xn0qmw74b9sqdoa/rotation_data.zip?raw=1).

### Dev cheatsheet
```bash
# Linting
make lint
# Build docker image for production
make build-docker
# Run production image API on port 8080
make run-api-docker
# Downloads all trained models into the /models folder
python -m codenames.preloader
# Various preprocessing options for images
# split: cuts images into 3 pieces, in order to reduce number of cards in each image
# extract_cards: extracts all codenames cards into separate images
# rotate: fixes card image rotations
python -m codenames.preprocess extract_cards --in ./raw_images --out ./processed_images
# Train a rotation model. Input images should have a 0 degree rotation
python -m codenames.train_rotation --in ./cards --out ./model.h5
# Prints all words found on images
python -m codenames.detect_words --in ./cards
```
