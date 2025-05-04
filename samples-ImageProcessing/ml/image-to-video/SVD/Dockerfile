FROM tensorflow/tensorflow:2.19.0-gpu

ARG USER_ID
ARG GROUP_ID

RUN apt-get update -qqy && \
    apt-get install -qqy build-essential python3-tk libxrender-dev libxext6 \
      libsm6 libglib2.0-0 libgl1-mesa-glx ffmpeg libgl1 libglib2.0-0 && \
    groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python

COPY src/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

USER python
