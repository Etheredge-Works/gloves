FROM python:3.7
RUN apt-get update && \
    pip install -r requrements.txt \
    && rm -rf /var/lib/apt/lists/
