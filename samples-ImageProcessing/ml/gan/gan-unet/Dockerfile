FROM tensorflow/tensorflow:latest-gpu

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends libgl1 libglib2.0-0 && \
    pip install Pillow opencv-python

USER python
