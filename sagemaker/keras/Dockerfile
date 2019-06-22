FROM jihong/keras-gpu:latest

RUN pip install futures flask gevent gunicorn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

# Set up the program in the image
COPY container /opt/program
WORKDIR /opt/program
