import redis
from custom_model import get_model
from custom_data import base64_decode
from settings import REDIS_PORT, REDIS_HOST, REDIS_QUEUE_NAME
import settings
import json
import numpy as np
IMAGE_JSON_NAME = 'image'
IMAGE_DTYPE = 'float32'
IMAGE_JSON_ID_NAME = 'id'
db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT)
model = None

def main():
    print("And so my watch begins.")
    model = get_model()
    while True:
        queue = db.lrange(name=REDIS_QUEUE_NAME, start=0, end=settings.BATCH_SIZE-1)
        image_ids = []
        batch = None

        for item in queue:
            json_item = json.loads(item.decode('utf-8'))
            image = base64_decode(item[IMAGE_JSON_NAME],
                                  IMAGE_DTYPE,
                                  (1, settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            image_ids.append(json_item[IMAGE_JSON_ID_NAME])
        if len(image_ids) > 0:
            print(f"Batch size; {batch.shape}")
            preds = model.predict(batch)
            results = imagenet_utils.decode_predictions(preds)
            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, results):
                # initialize the list of output predictions
                output = []
                # loop over the results and add them to the list of
                # output predictions
                for (imagenetID, label, prob) in resultSet:
                    r = {"label": label, "probability": float(prob)}
                    output.append(r)
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))
            # remove the set of images from our queue
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)


