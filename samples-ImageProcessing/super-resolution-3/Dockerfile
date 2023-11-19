FROM python:3.6.15-slim-buster

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends libgl1 libglib2.0-0 && \
    pip install ISR && \
    pip install 'h5py==2.10.0' --force-reinstall && \
    mkdir /app

USER python
WORKDIR /app
