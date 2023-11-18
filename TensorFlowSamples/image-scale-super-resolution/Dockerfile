FROM tensorflow/tensorflow:latest

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    pip install tensorflow-hub

USER python
