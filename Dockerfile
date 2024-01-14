# Image: jstubbs/coe379l
# This image can be used to run jupyter notebook server and execute the class examples. 
# 
# Note: To build this image, first export a value for the ENV variable, e.g., 
#    export ENV=production; docker build -t jstubbs/coe379l .

from python:3.11

# the environment, should be either "dev" or "production"
ARG ENV

ENV ENV=${ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.7.0

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/


# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install $(test "$ENV" == production && echo "--no-dev") --no-interaction --no-ansi

# Creating folders, and files for a project:

