FROM python:3.6.7-alpine3.6

ADD clean-layer.sh  /tmp/clean-layer.sh
RUN chmod +x /tmp/clean-layer.sh

RUN pip install install mxnet-mkl --pre && /tmp/clean-layer.sh

# Set up the program in the image
COPY container /opt/program
WORKDIR /opt/program





