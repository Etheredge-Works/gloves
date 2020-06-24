#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf 2.1
FROM tensorflow/tensorflow:latest-gpu
COPY . /app
WORKDIR /app
RUN apt-get update && \
    pip install -r requirements.txt
#RUN dvc repro dvc/split
#RUN dvc repro dvc/split

