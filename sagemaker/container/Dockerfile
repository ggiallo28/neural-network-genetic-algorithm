FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Andreas Fritzler <andreas.fritzler@gmail.com>

RUN pip --no-cache-dir install keras

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update && apt-get install -qq -y \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/fchollet/keras.git /keras

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"

RUN pip install futures flask gevent gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

# Set up the program in the image
COPY reinf /opt/program
WORKDIR /opt/program
