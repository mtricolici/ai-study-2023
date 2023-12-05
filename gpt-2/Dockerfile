#FROM tensorflow/tensorflow:2.15.0
#FROM tensorflow/tensorflow:1.12.0-py3
FROM tensorflow/tensorflow:1.15.5-py3

ENV LANG=C.UTF-8

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    mkdir /gpt2

ADD gpt2-src/requirements.txt /gpt2/requirements.txt

WORKDIR /gpt2

RUN pip install -r requirements.txt

USER python
