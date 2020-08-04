#!/bin/bash -e
full_image_name=etheredgeb/gloves:pipeline
docker run -v $(pwd):/pipeline $full_image_name python /pipeline/src/pipeline.py