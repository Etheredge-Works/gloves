FROM tensorflow/tensorflow:2.7.0-gpu
COPY docker-requirements.txt /tmp/requirements.txt

# RUN apt update \
#     && apt install -y git \ 
    # && pip3 install -r /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt