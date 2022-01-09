FROM tensorflow/tensorflow:2.7.0-gpu
COPY docker-requirements.txt /tmp/requirements.txt

# RUN apt update \
#     && apt install -y git \ 
    # && pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir /app && chmod 777 /app

# wandb complains about this file for some reason
RUN mkdir /.config && chmod 777 /.config

WORKDIR /app