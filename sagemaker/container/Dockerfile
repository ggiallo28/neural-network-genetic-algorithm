FROM jihong/keras-gpu:latest

RUN pip install futures flask gevent gunicorn
RUN pip install --ignore-installed --upgrade "https://github.com/lakshayg/tensorflow-build/releases/download/v1.6.0-ubuntu_16_04-py27-avx-avx2-fma-sse41-sse42/tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl" --user

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

# Set up the program in the image
COPY reinf /opt/program
WORKDIR /opt/program
