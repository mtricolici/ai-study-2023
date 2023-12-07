#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM tensorflow/tensorflow:2.13.0-gpu

ARG USER_ID
ARG GROUP_ID
ENV DEBIAN_FRONTEND=noninteractive

RUN groupadd -g $GROUP_ID python && \
    useradd -m -u $USER_ID -g $GROUP_ID python && \
    \
    rm -f /etc/apt/sources.list.d/cuda.list && \
    \
    apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends \
      libgl1-mesa-glx libglib2.0-0 libxrender-dev libxext6 \
      ffmpeg build-essential

ADD requirements.txt /

# Uncomment some mirror when fucking pypi is down :@
#RUN pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#RUN pip config --user set global.index-url https://mirror.ox.ac.uk/sites/pypi.python.org/simple/
RUN pip config --user set global.index-url https://mirrors.sustech.edu.cn/pypi/web/simple

RUN pip install -r /requirements.txt

ADD nafnet /nafnet

# This library just use print without possiblity to hide warnings!!!
RUN find /usr/local/lib/python3.8/dist-packages/insightface -name "*.py" -type f -exec sed -i 's/print(/pass#print(/g' {} +

USER python
