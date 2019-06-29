FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ADD clean-layer.sh  /tmp/clean-layer.sh
RUN chmod +x /tmp/clean-layer.sh

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && \
    apt-get update && apt-get install -y --no-install-recommends python3.6-dev && \
    ln -s -f /usr/bin/python3 /usr/bin/python && \
    apt-get install -y --no-install-recommends python3-distutils && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && rm get-pip.py 
RUN rm /usr/local/bin/pip && ln -s /usr/local/bin/pip3 /usr/local/bin/pip
RUN apt-get remove curl -y && /tmp/clean-layer.sh
RUN pip install mxnet-cu101mkl && /tmp/clean-layer.sh

# Set up the program in the image
COPY container /opt/program
WORKDIR /opt/program




        
        