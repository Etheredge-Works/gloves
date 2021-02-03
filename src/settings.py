import tensorflow as tf

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
BATCH_SIZE = 8
TEST_RATIO = 0.2
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
DATA_DIR = 'data/images'
#AUTOTUNE = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE
DENSE_NODES = 32
EPOCHS = 1
MIXED_PRECISION = False  # TODO for some reason this cuts memory usage from > 11GB to like 1GB....
DOGS = [

]
CATS = [

]
REDIS_PORT = 6379
REDIS_HOST = "db"
REDIS_QUEUE_NAME = 'queue'