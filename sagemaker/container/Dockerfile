# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get install -y --no-install-recommends wget nginx ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow-gpu keras futures flask gevent gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

# Set up the program in the image
COPY reinf /opt/program
WORKDIR /opt/program
