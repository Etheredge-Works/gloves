import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import pathlib
import os
import streamlit as st
import numpy as np
import re
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
DATA_DIR = "images"
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

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
    #split_anchor_path = tf.strings.split(anchor_file_path, sep=os.path.sep)
    #anchor_directory = tf.strings.join(split_anchor, sep=os.path.sep)
    # TODO tweak random var
    #3random_value = tf.random.uniform((0,), minval=minval, maxval=1)
    #random_value = np.random.randint(-1, 1)
    # TODO might have two of the anchor, but oh well
    #positives = list(pathlib.Path(anchor_directory).glob(f"{anchor_label}*"))
    # TODO make more efecient
    #global ALL_FILES
    #all_files = ALL_FILES
    #all_files = copy.deepcopy(ALL_FILES) # TODO find a faster way
    #match
    #all_files.remove(os.path.basename(anchor_file_path))
    #all_files.remove(split_anchor_path[-1])
    #converted = str(anchor_label)

    labels = tf.strings.regex_replace(ALL_FILES, pattern=r'_\d.*', rewrite='')
    #3labels = tf.strings.regex_replace(ALL_FILES, pattern='\..*', rewrite='')
    #file_name = tf.strings.split(ALL_FILES, sep=os.path.sep)[-1]
    #print(f"labels: {labels}")
    pos_mask = anchor_label == labels
    #pos_mask = tf.strings.regex_full_match(ALL_FILES, fr'{anchor_label}_\d+')

    label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)
    #label = tf.cast(tf.math.round(tf.random.uniform([], 0, 1)), dtype=tf.int32)

    mask = tf.cond(label == tf.constant(1), lambda: pos_mask, lambda: tf.math.logical_not(pos_mask))
    #mask = tf.cond(label == tf.constant(1), lambda: pos_mask, lambda: tf.math.logical_not(pos_mask))
    #if to_flip:
         #mask = tf.math.logical_not(pos_mask)
    #else:
         #mask = pos_mask

    #pos = tf.boolean_mask(pos_mask, all_files)
    #pos = tf.boolean_mask(all_files, pos_mask)
    #neg = tf.boolean_mask(tf.math.logical_not(pos_mask), all_files)
    #neg = tf.boolean_mask(all_files, tf.math.logical_not(pos_mask))
    values = tf.boolean_mask(ALL_FILES, mask)
    #tf.print(anchor_file_path, "file_path")
    #tf.print(anchor_label, "label")
    #Etf.assert_greater(tfI.size(anchor_label), 0)
    #Etf.assert_greater(tf.size(values), 0, "")

    #test = tf.strings.split(all_files, sep='_')[:, 0]I#
    # TODO linearize
    '''
    for file, splits in zip(all_files, labels):
        #3label = tf.strings.split(file, sep='_')[0]
        #label = splits[0]
        label = splits
        if label == anchor_label:
            pos.append(file)
        else:
            neg.append(file)
    '''
    #if len(pos) == 0 or len(neg) == 0:
        #print("well")
    #negatives = list(set(all_files) - set(positives))

    tf.debugging.assert_greater(tf.size(values), tf.constant(0), f"Values are empty.\n")
    #tf.debugging.assert_equal(tf.shape(values), tf.constant(1), f"Size is wrong.\n")
    idx = tf.random.uniform([], 0, 2, dtype=tf.int32)
    #idx = tf.random.uniform([], 0, tf.size(values), dtype=tf.int32)
    #idx = tf.random.uniform([], 0, len(values), dtype=tf.int32)
    #neg_idx = tf.random.uniform([], 0, len(neg), dtype=tf.int32)
    #pos_idx = tf.random.uniform([], 0, len(pos), dtype=tf.int32)
    #pos_idx = tf.random.uniform([1], 0, len(pos), dtype=tf.int32)
    #neg_idx = np.random.randint(0, len(neg))
    #pos_idx = np.random.randint(0, len(pos))
    #negative = tf.reshape(tf.gather(neg, tf.stack((neg_idx))), shape=[])
    #positive = tf.reshape(tf.gather(pos, tf.stack((pos_idx))), shape=[])
    #positive = tf.reshape(tf.gather(pos, tf.stack((pos_idx))), shape=[])
    value = tf.gather(values, idx)
    #print(negative)
    #print(positive)
    #positive = pos[np.random.randint(0, len(pos))]
    #positive = pos[np.random.randint(0, len(pos))]
    #anchor_directory = pathlib.Path(anchor_directory)
    path = tf.strings.join([TF_DATA_DIR, value], os.path.sep)
    sq_path = tf.squeeze(path)
    return anchor_file_path, sq_path, label
    #return anchor_file_path, tf.reshape(tf.strings.join([anchor_directory, positive], os.path.sep), shape=[]), \
           #tf.reshape(tf.strings.join([anchor_directory, negative], os.path.sep), shape=[])

    #negatvie = negatives[]

    if random_value < -1.33333:
        # get easy
        pass

    elif random_value < -1.66666666666:
        # get semi_hard
        pass
    else:
        # get_hard
        pass


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


