FROM tensorflow/tensorflow:2.7.0-gpu
COPY requirements.txt /tmp/requirements.txt

# RUN apt update \
#     && apt install -y git \ 
    # && pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt \
    && mkdir /app && chmod 777 /app \
    && apt update \
    && apt install -y git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /.config && chmod 777 /.config
# wandb complains .config for some reason

WORKDIR /app