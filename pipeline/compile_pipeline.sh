#!/bin/bash -e
full_image_name=etheredgeb/gloves:pipeline
touch ~/.env
docker run --env-file ../.devcontainer/.env -v $(pwd):/pipeline $full_image_name python /pipeline/src/pipeline.py