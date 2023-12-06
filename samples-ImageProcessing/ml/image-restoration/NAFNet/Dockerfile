FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 as cuda

RUN mkdir -p /TO-COPY
RUN find / \
    -not \( -path "/sys" -prune \) \
    -not \( -path "/proc" -prune \) \
    -not \( -path "/dev" -prune \) \
    -not \( -path "/TO-COPY" -prune \) \
    -type f \( -path "*cuda*" -o -path "*nvidia*" \) \
    -exec cp --parents {} /TO-COPY \;
#    > /TO-COPY/cuda.txt

FROM python:3.9.5-slim

ARG USER_ID
ARG GROUP_ID

COPY --from=cuda /TO-COPY/ /

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python

ADD requirements.txt /

RUN pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch==1.11.0
RUN pip install -r /requirements.txt

USER python
