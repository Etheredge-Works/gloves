#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf 2.1
FROM tensorflow/tensorflow:latest-gpu
COPY . /app
WORKDIR /app
RUN apt-get update && \
    apt-get install -y git wget && \
    pip install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/
#RUN dvc repro dvc/split
#RUN dvc repro dvc/split

