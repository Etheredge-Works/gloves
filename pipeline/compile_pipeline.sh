#!/bin/bash -e
full_image_name=etheredgeb/gloves_kubeflow:latest
touch ../.devcontainer/.env  #ensure file exists
# TODO correct path
docker run --env-file ../.devcontainer/.env -v $(pwd):/pipeline $full_image_name python /pipeline/src/pipeline.py