#FROM python:3.7
# Starting from tf docker image is the easiest way to get tf >2.1 and gpu libraries
#FROM tensorflow/tensorflow:2.4.1-gpu
FROM tensorflow/tensorflow:2.2.2-gpu
# TODO why does 2.4 break printing metrics from fit

ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# for now, I just want the latest images
#COPY requirements.txt params.yaml /app/
#RUN pip install -r /app/requirements.txt
RUN pip install mlflow wandb click tensorflow_addons==0.11.0 boto3 icecream scikit-learn dvc dvclive
RUN pip install siamese==0.0.34 
COPY . /gloves
WORKDIR /gloves
ENTRYPOINT ["python"]