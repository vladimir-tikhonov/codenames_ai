FROM python:3.7-alpine

WORKDIR /opt/codenames_ai

RUN apk --update add --no-cache \
    lapack-dev \
    gcc \
    freetype-dev \
    gfortran \
    musl-dev \
    g++

RUN pip install pipenv
COPY Pipfile /opt/codenames_ai/Pipfile
COPY Pipfile.lock /opt/codenames_ai/Pipfile.lock
RUN pipenv install --system

COPY codenames /opt/codenames_ai/codenames
COPY config.ini /opt/codenames_ai/config.ini

RUN python -m codenames.preloader

CMD ["waitress-serve", "--call", "codenames.api:create_app"]
