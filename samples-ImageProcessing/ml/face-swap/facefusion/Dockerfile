#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM tensorflow/tensorflow:2.13.0-gpu

ARG USER_ID
ARG GROUP_ID
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    \
    apt-get update -qqy && \
    \
    apt-get install -qqy --no-install-recommends \
      python3.10 \
      python-is-python3 \
      pip \
      git \
      curl \
      ffmpeg && \
    apt-get clean -qqy

RUN rm -f /opt/nvidia/entrypoint.d/*copyright.txt /opt/nvidia/entrypoint.d/*license.txt

ADD src/requirements.txt /

RUN pip install -r /requirements.txt && \
    rm -rf /root/.cache

USER python
