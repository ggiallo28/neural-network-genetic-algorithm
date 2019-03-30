FROM jihong/keras-gpu:latest

RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6

RUN pip install futures flask gevent gunicorn
RUN pip install --ignore-installed --upgrade "https://github.com/sigilioso/tensorflow-build/raw/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl" --user

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

# Set up the program in the image
COPY reinf /opt/program
WORKDIR /opt/program
