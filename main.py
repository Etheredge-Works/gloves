#! python
import os

from data import CatDogData

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import streamlit as st
import numpy as np

from utils import euclidean_distance

np.random.seed(4)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#st.title("Mittens and Dave similarity analaysis")
import utils
import tensorflow as tf

tf.random.set_seed(4)

MIXED_PRECISION = False
if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

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

# %%

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

#model = build_model()

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
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Input, Lambda

input_shape = (utils.IMG_WIDTH, utils.IMG_HEIGHT, 3)
#input = Input(input_shape)
print(tf.__version__)


class Encoder(tf.keras.Model):
    def __init__(self, should_transfer_learn=False):
        super(Encoder, self).__init__()

        #model_input = Input(input_shape)
        #x = model_input
        self.model_layers = [
            Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='same', activation='relu', use_bias=False),
            Conv2D(filters=64, kernel_size=(6, 6), strides=3, padding='same', activation='relu', use_bias=False),
            Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', activation='relu', use_bias=False),
            Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
            Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
            #Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', use_bias=False),
            Flatten(),
            Dense(16, activation='relu'),
            #Dropout(0.1),
        ]

    def call(self, inputs):
        #tf.print(inputs, "inputs")
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x


class GlovesNet(tf.keras.Model):
    def __init__(self, should_transfer_learn=False):
        super(GlovesNet, self).__init__()
        if should_transfer_learn:
            base_model = ResNet50V2(weights='imagenet',
                                     include_top=False,
                                     input_shape=input_shape)
            for layer in base_model.layers:
                layer.trainable = False
            x = base_model.output
            x = Flatten()(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.5)(x)
            self.encoder_model = Model(base_model.inputs, x)

        else:
            self.encoder_model = Encoder(should_transfer_learn=False)

        #self.input1 = Input(input_shape)
        #self.input2 = Input(input_shape)
        #print(type(self.input1))
        #print(type(self.encoder_model))

        #self.dense1 = self.encoder_model(self.input1)
        #self.dense2 = self.encoder_model(self.input2)
        #$self.dense1 = self.encoder_model(self.input1)
        #$self.dense2 = self.encoder_model(self.input2)

        self.merge_layer = Lambda(euclidean_distance) # ([self.dense1, self.dense2])
        self.prediction_layer = Dense(1, activation="sigmoid") # (self.merge_layer)
        #model = Model(inputs=[input1, input2], outputs=dense_layer)

    def call(self, inputs):
        #tf.print(inputs)
        x1, x2 = inputs
        encoding1 = self.encoder_model(x1)
        encoding2 = self.encoder_model(x2)
        merged = self.merge_layer([encoding1, encoding2])
        prediction = self.prediction_layer(merged)
        return prediction
        #self.add_loss()

        #model.compile(loss = "binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
        #model.compile(loss = "binary_crossentropy", optimizer='adam', metrics=["accuracy"])
        #model.summary()

model = GlovesNet()
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])

train_ds, test_ds = CatDogData.data()
'''
for (anchor, other), label in train_ds.take(32):
    #st.write(f.numpy())
    print(anchor.numpy())
    print(other.numpy())
    print(label.numpy())
'''

#wandb.init(project="siamese")
#model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=128, epochs=20, callbacks=[WandbCallback()])
model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds,
    validation_steps=np.floor((CatDogData.test_set_size) / utils.BATCH_SIZE),
    validation_freq=1,
    steps_per_epoch=CatDogData.STEPS_PER_EPOCH)
#validation_data = ([pairs_test[:, 0], pairs_test[:, 1]], labels_test[:]))


