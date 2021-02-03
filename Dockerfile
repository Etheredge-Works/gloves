#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf 2.1
FROM tensorflow/tensorflow:latest-gpu
COPY requirements.txt params.yaml /app/
#RUN pip install -r /app/requirements.txt
RUN pip install mlflow wandb click tensorflow_addons boto3
COPY src /app/src
WORKDIR /app