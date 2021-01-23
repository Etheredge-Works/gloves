#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf 2.1
FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt params.yaml /app/
COPY src /app/src
WORKDIR /app
RUN pip install -r requirements.txt