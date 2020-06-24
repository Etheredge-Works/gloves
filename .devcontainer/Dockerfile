#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf 2.1
FROM tensorflow/tensorflow:latest-gpu
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y wget git && \
    pip install -r requirements.txt
RUN mkdir data && dvc repro dvc/split_data.dvc
#RUN dvc repro dvc/split

