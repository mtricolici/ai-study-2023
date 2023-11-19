FROM tensorflow/tensorflow:latest

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends \
      python3-sdl2 && \
    pip install pygame

USER python
