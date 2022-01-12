#! /bin/bash
docker run -it --rm -v $PWD:/app \
    -e "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" \
    -e "MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL" \
    -e MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME \
    -e MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    --user $(id -u):$(id -g) \
    --gpus all \
    etheredgeb/gloves:custom-tensorflow-2.7.0 \
    "$@"