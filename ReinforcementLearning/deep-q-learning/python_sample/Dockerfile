FROM tensorflow/tensorflow:latest-gpu
ARG USER_ID
ARG GROUP_ID
RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    apt-get install -qqy x11-xserver-utils x11-utils && \
    pip install pygame

USER python
