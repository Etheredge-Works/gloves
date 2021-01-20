#!/bin/bash -e
full_image_name=etheredgeb/gloves:pipeline
touch ../.devcontainer/.env  #ensure file exists
docker run --env-file ../.devcontainer/.env -v $(pwd):/pipeline $full_image_name python /pipeline/src/pipeline.py