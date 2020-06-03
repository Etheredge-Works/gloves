import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import backend as K
import pathlib
import os
import streamlit as st
import numpy as np
import re

def limit_gpu_memory_use():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

limit_gpu_memory_use()

ALL = [
    'Abyssinian',
    'yorkshire_terrier',
    'american_bulldog',
    'american_pit_bull_terrier'
]

DOGS = [

]

CATS = [

]


def get_dataset(dir="images", url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"):
    raw_data_dir = pathlib.Path(get_file(dir, url, untar=True))
    # remove dumb mat files
    # TODO handle mat files better
    mat_files = raw_data_dir.glob("*mat")
    for file in mat_files:
        os.remove(file)
    return raw_data_dir

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 64
TEST_RATIO = 0.2
AUTOTUNE = tf.data.experimental.AUTOTUNE
DATA_DIR = get_dataset()
TF_DATA_DIR = tf.constant(str(DATA_DIR))
CLASS_NAMES = np.array(list(set(
    [re.sub(r'_\d.*', '', item.name)
     for item in DATA_DIR.glob('*.jpg') if item.name != "LICENSE.txt"])))

import copy

# TODO could change to glob of jpgs
ALL_FILES = tf.io.gfile.listdir(str(DATA_DIR))
#print(f"all_files: {ALL_FILES}")
def get_pairs(anchor_file_path, minval=tf.constant(-1)):
    anchor_label = get_label(anchor_file_path)
    # TODO tweak random var
    # TODO might have two of the anchor, but oh well
    # TODO make more efecient

    split_path = tf.strings.split(anchor_file_path, sep=os.path.sep)
    file_name = tf.gather(split_path, tf.size(split_path) - 1)
    anchor_file_mask = ALL_FILES == file_name

    labels = tf.strings.regex_replace(ALL_FILES, pattern=r'_\d.*', rewrite='')
    pos_mask = anchor_label == labels

    label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)

    pos_label_func = lambda: tf.math.logical_xor(pos_mask, ALL_FILES == file_name) # XOR prevents anchor file being used
    neg_label_func = lambda: tf.math.logical_not(pos_mask)
    mask = tf.cond(label == tf.constant(1), pos_label_func, neg_label_func)
    values = tf.boolean_mask(ALL_FILES, mask)
    '''
    # TODO implement a way to grab easy, medium, hard pairs (e.g. boxer-cat, boxer-dog, boxer-pug or with losses)
    # TODO Need a way to monitor losses and let it inform seletion weights
    if random_value < -1.33333:
        # get easy
        pass

    elif random_value < -1.66666666666:
        # get semi_hard
        pass
    else:
        # get_hard
        pass
    '''

    tf.debugging.assert_greater(tf.size(values), tf.constant(0), f"Values are empty.\n")
    idx = tf.random.uniform([], 0, 2, dtype=tf.int32)
    value = tf.gather(values, idx)
    path = tf.strings.join([TF_DATA_DIR, value], os.path.sep)
    sq_path = tf.squeeze(path)
    return anchor_file_path, sq_path, label


def get_label(file_path):
    #assert (len(str(file_path)) > 0)
    #st.write(file_path)
    # Get file name
    #file_path = str(file_path)
    #tf.print(file_path, "file_path")
    file_name = tf.strings.split(file_path, sep=os.path.sep)[-1]
    #tf.print(file_name, "file_name")
    # label = tf.strings.regex_replace(file_path, f'.*{os.path.sep}', '')

    # Strip down to classname
    label = tf.strings.regex_replace(file_name, r'_\d+\.jpg.*', '')
    # TODO can use digit to make sure anchor and positive are different
    #assert (len(str(label)) > 0)
    #st.write(f"label: {label}")
    return label

def get_encoded_label(file_path):
    return get_label(file_path) == CLASS_NAMES


def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: tf.image.random_crop(x, size=(IMG_HEIGHT, IMG_WIDTH, 3)))


def simple_decode(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = preprocess_input(img)  # NOTE: This does A TON for accuracy
    #img = tf.image.convert_image_dtype(img, tf.float32)  #TODO remove this if preproces sis used
    return img


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor

    # img = tf.image.decode_jpeg(img)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.decode_image(img)
    # img = tf.image.decode_jpeg(img)
    # img = tf.image.decode_image(img, channels=0)
    # img = tf.image.decode_jpeg(img, channels=0)
    # img = tf.image.resize(img, [IMG_WIDTH*2, IMG_HEIGHT*2])
    # img = zoom(img)

    # st.write(img)

    # img = tf.image.convert_image_dtype(img, tf.float32)

    NUM_BOXES = 4
    boxes = tf.random.uniform(shape=(NUM_BOXES, 3))
    box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=BATCH_SIZE, dtype=tf.int32)
    # img = tf.image.crop_and_resize(img, boxes, box_indices, (IMG_HEIGHT, IMG_WIDTH))
    # st.write(img)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # resize the image to the desired size.
    # img = tf.image.random_crop(img, [IMG_WIDTH, IMG_HEIGHT, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)

    # img = tf.image.convert_image_dtype(img, tf.float16)

    # img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = preprocess_input(img)  # This handles float conversion
    # img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    #img = tf.image.convert_image_dtype(img, tf.float32)  #TODO remove this if preproces sis used

    # st.write(img)
    return img


def read_images(decode_func, test_decode):
    def foo(file_path):
        #label = utils.get_encoded_label(file_path)
        anchor_file_path, other_file_path, label = get_pairs(file_path)
        # TODO may not need to check for different anchor/positive since they'll get morephed differently...
        # load the raw data from the file as a string
        #print(anchor_file_path)
        #jprint(positive_file_path)
        #print(negative_file_path)
        anchor_file = tf.io.read_file(anchor_file_path)
        other_file = tf.io.read_file(other_file_path)
        #negative_file = tf.io.read_file(negative_file_path)

        #anc_img = decode_func(anchor_file) # TODO maybe do no encoding to anc
        anc_img = test_decode(anchor_file) # TODO maybe do no encoding to anc
        #other_img = decode_func(other_file)
        other_img = test_decode(other_file)
        #neg_img = decode_func(negative_file)
        #return anc_img, label
        return (anc_img, other_img), label

    return foo


def prepare_for_training(ds, cache=False, shuffle=True, shuffle_buffer_size=BATCH_SIZE):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

