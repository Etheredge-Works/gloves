#! python
import streamlit as st
import numpy as np
np.random.seed(4)
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#st.title("Mittens and Dave similarity analaysis")
import tensorflow as tf
tf.random.set_seed(4)
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

MIXED_PRECISION = False
if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)
import utils
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from tensorflow.keras.applications.resnet_v2 import ResNet152V2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#assert tf.multiply(6, 7).numpy() == 42

#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

#st.write('Compute dtype: %s' % policy.compute_dtype)
#st.write('Variable dtype: %s' % policy.variable_dtype)


raw_data_dir = utils.get_dataset()

import pathlib
data_dir = pathlib.Path(raw_data_dir)
data_dir

image_count = len(list(data_dir.glob('*.jpg')))
real_count = image_count - int(image_count * utils.TEST_RATIO)
image_count

import numpy as np
import re
#CLASS_NAMES

import IPython.display as display
from PIL import Image

mittens_knockoffs = list(data_dir.glob('Maine_Coon*'))
dave_knockoffs = list(data_dir.glob('boxer*'))

for image_path in mittens_knockoffs[:3]:
    st.image(Image.open(str(image_path)))
    display.display(Image.open(str(image_path)))
for image_path in dave_knockoffs[:3]:
    st.image(Image.open(str(image_path)))
    display.display(Image.open(str(image_path)))

# %%

import tensorflow as tf

st.write(str(data_dir / '*.jpg'))
all_ds = tf.data.Dataset.list_files(str(data_dir / '*.jpg'))
test_size = int(image_count * utils.TEST_RATIO)
# test_size = np.ceil(image_count * TEST_RATIO)
test_ds = all_ds.take(test_size)
list_ds = all_ds.skip(test_size)

# %%

str(data_dir / '*.jpg')

# %%

for f in list_ds.take(32):
    st.write(f.numpy())

# %%

import os





# %%

# st.write(CLASS_NAMES)
for f in list_ds.take(10):
    st.write()
    st.write(utils.get_label(f))


# %%


# %%

#generator = tf.keras.preprocessing.image.ImageDataGenerator()


def read_images(decode_func, test_decode):
    def foo(file_path):
        #label = utils.get_encoded_label(file_path)
        anchor_file_path, other_file_path, label = utils.get_pairs(file_path)
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

# %%

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
# labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_ds = list_ds.map(read_images(utils.decode_img, utils.simple_decode), num_parallel_calls=utils.AUTOTUNE)

test_labeled_ds = test_ds.map(read_images(utils.simple_decode, utils.simple_decode), num_parallel_calls=utils.AUTOTUNE)

# %%

'''
for image, label in labeled_ds.take(10):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
'''

# %%


#st.write(CLASS_NAMES)
#for f in list_ds.take(1):
    #labeled = utils.process_path(f)
    #image, label = labeled
    # st.write("Image shape: ", image.numpy().shape)
    # st.write("Label: ", label)


# %%

# def prepare_for_training(ds, cache=True, shuffle=True, shuffle_buffer_size=1000):
def prepare_for_training(ds, cache=False, shuffle=True, shuffle_buffer_size=real_count):
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

    ds = ds.batch(utils.BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=utils.AUTOTUNE)

    return ds


# %%

st.write("Prepare training")
# train_ds = prepare_for_training(labeled_ds)
train_ds = prepare_for_training(labeled_ds)

# print("Prepare testing")
test_ds = prepare_for_training(test_labeled_ds, shuffle=False)

st.write("batch test")
item = next(iter(train_ds))
item = next(iter(train_ds))

#(image_batch, img_batch2), label_batch = next(iter(train_ds))
image_batch, label_batch = next(iter(train_ds))

#(image_batch, img_batch2), label_batch = next(iter(train_ds))
#(test_image_batch, _), test_label_batch = next(iter(test_ds))
st.write("done")

# %%

#CLASS_NAMES

import matplotlib.pyplot as plt



'''
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    print(f"image_batch: {image_batch[0].shape}")
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[0][n])
        # print(label_batch[n])
        #plt.title(label_batch[n])
        plt.axis('off')


print("showing batches")
show_batch(image_batch, label_batch)
#show_batch(test_image_batch, test_label_batch)
'''
# %%

import time

default_timeit_steps = 1000


def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            st.write('.', end='')
    st.write()
    end = time.time()

    duration = end - start
    st.write("{} batches: {} s".format(steps, duration))
    st.write("{:0.5f} Images/s".format(utils.BATCH_SIZE * steps / duration))


# %%

# `tf.data`
# timeit(train_ds)

# %%

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, MaxPooling2D

# %%

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

#model = build_model()

test_set_size = np.floor(image_count * utils.TEST_RATIO)
STEPS_PER_EPOCH = np.ceil((image_count - test_set_size) / utils.BATCH_SIZE)
# STEPS_PER_EPOCH = np.ceil((image_count-test_set_size)/BATCH_SIZE) * 5
# test_dataset = train_ds.take(test_set_size
# train_dataset = train_ds.skip(test_set_size)

'''
model.fit(
    train_ds,
    epochs=50,
    validation_data=test_ds,
    validation_steps=np.ceil((test_set_size) / BATCH_SIZE),
    # validation_steps=np.floor((test_set_size)/BATCH_SIZE),
    validation_freq=1,
    steps_per_epoch=STEPS_PER_EPOCH,
)
'''

# %%
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Lambda

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

input_shape = (utils.IMG_WIDTH, utils.IMG_HEIGHT, 3)
#input = Input(input_shape)
print(tf.__version__)


transfer_learning = False
if transfer_learning:
    base_model = ResNet50V2(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    dense = Model(base_model.inputs, x)

else:
    model_input = Input(input_shape)
    x = model_input
    x = Conv2D(filters=32, kernel_size=(8,8), strides=4, padding='same', activation='relu', use_bias=False)(x)
    x = Conv2D(filters=64, kernel_size=(6,6), strides=3, padding='same', activation='relu', use_bias=False)(x)
    x = Conv2D(filters=128, kernel_size=(4,4), strides=2, padding='same', activation='relu', use_bias=False)(x)
    x = Conv2D(filters=256, kernel_size=(2,2), strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False)(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=2, padding='same', activation='relu', use_bias=False)(x)
    x = Flatten()(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.1)(x)
    dense = Model(model_input, x)

#x = Conv2D(filters=512, kernel_size=(7,7), strides=4, padding='same', activation='relu', use_bias=False)(x)
dense.summary()

input1 = Input(input_shape)
input2 = Input(input_shape)

dense1 = dense(input1)
dense2 = dense(input2)

merge_layer = Lambda(euclidean_distance)([dense1,dense2])
dense_layer = Dense(1, activation="sigmoid")(merge_layer)
model = Model(inputs=[input1, input2], outputs=dense_layer)


model.compile(loss = "binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
#model.compile(loss = "binary_crossentropy", optimizer='adam', metrics=["accuracy"])
model.summary()


#wandb.init(project="siamese")
#model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=128, epochs=20, callbacks=[WandbCallback()])
model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    validation_steps=np.floor((test_set_size)/utils.BATCH_SIZE),
    validation_freq=1,
    steps_per_epoch=STEPS_PER_EPOCH)
#validation_data = ([pairs_test[:, 0], pairs_test[:, 1]], labels_test[:]))


