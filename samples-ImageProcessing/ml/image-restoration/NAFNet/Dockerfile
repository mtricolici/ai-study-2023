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

RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends libgl1-mesa-glx libglib2.0-0 ffmpeg

ADD requirements.txt /

# Uncomment some mirror when fucking pypi is down :@
#RUN pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip config --user set global.index-url https://mirror.ox.ac.uk/sites/pypi.python.org/simple/
RUN pip config --user set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple

RUN pip install -r /requirements.txt

ADD nafnet /nafnet

RUN cd /nafnet && python setup.py develop --no_cuda_ext

USER python
