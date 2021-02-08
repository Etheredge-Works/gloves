#from src import utils
from . import utils
utils.limit_gpu_memory_use()

import tensorflow as tf
from tensorflow.keras import Model
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as pre_trained_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as pre_trained_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np
#from wandb.keras import WandbCallback
import mlflow
mlflow.tensorflow.autolog(every_n_iter=1)

import settings
import pathlib

from utils import euclidean_distance, input_shape, get_dataset_values


class Encoder(tf.keras.Model):
    def __init__(self,
                 dense_nodes=settings.DENSE_NODES):
        super(Encoder, self).__init__()

        #model_input = Input(input_shape)
        #x = model_input


    def call(self, inputs):
        #tf.print(inputs, "inputs")
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x


# Using functional API instead of classes as it allows for saving of model and weights together
def build_model(should_transfer_learn=False):
    pass


def weight_init():
    return tf.random_normal_initializer(mean=0, stddev=0.01)


def bia_init():
    return tf.random_normal_initializer(mean=0.5, stddev=0.01)

def reg():
    #return None
    return l2(2e-4)


#def contrastive_loss(vects):
    #margin = 1
    #square_pred = K.square(y_pred)
    #margin_square = K.square(K.maximum(margin - y_pred, 0))
    #return K.mean(y_true * square_pred + (1 - y_true) * margin_square

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

class NWayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            #predictions = [self.model.predict_on_batch(n_way_batch) for n_way_batch in test_ds]
            predictions = model.predict(test_ds)
            highest_predictions = [np.argmax(prediction) for prediction in predictions]
            correct_predictions = [highest_prediction == 0 for highest_prediction in highest_predictions]
            score = np.average(correct_predictions)
            print(f"\n\nN-Way Accuracy: {score}\n")


def save_model(model):
    model.save("model")


def get_model():
    #net = glovesnet(should_transfer_learn=True)
    #net.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
    return tf.keras.models.load_model("model.h5")
    #net.load_weights("model")
    #return net
    # TODO make model name variable


def run_model():
    model = tf.keras.models.load_model("model")
    train_ds, test_ds, steps_per_epoch, validation_steps = get_dataset_values(
        utils.get_dataset(),
        settings.BATCH_SIZE,
        settings.TEST_RATIO)
    score = model.evaluate(test_ds, steps=validation_steps)
    return score


##############################################################################
class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(DistanceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        return euclidean_distance(inputs)

def log_model(func):
    def wrapper(*args, **kwargs):
        #model = func(*args, **kwargs)
        model: tf.keras.Model = func(*args, **kwargs)
        log_summary(model)
        return model
    return wrapper
    
def log_summary(model):
    filename = model.name + ".txt"
    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    mlflow.log_artifact(filename)
@log_model
def combine_models(base_model, head_model, name="no_name"):
    #print(base_model.input_shape)
    #print(base_model.output_shape)
    #print(head_model.input_shape)
    #print(head_model.output_shape)
    encoder_inputs = tf.keras.Input(base_model.input_shape[1:]), tf.keras.Input(base_model.input_shape[1:])
    encoder_outputs = head_model([base_model(encoder_inputs[0]), base_model(encoder_inputs[1])])
    return Model(name=name, inputs=encoder_inputs, outputs=encoder_outputs)

@log_model
def distance_model(input_shape):
    input1 = tf.keras.Input(input_shape)
    input2 = tf.keras.Input(input_shape)
    y_pred = DistanceLayer(dtype='float32')([input1, input2])
    return Model(inputs=(input1, input2), outputs=y_pred, name='distance_model')

@log_model
def sigmoid_model(input_shape):
    input1 = tf.keras.Input(input_shape)
    input2 = tf.keras.Input(input_shape)
    x = Concatenate(dtype='float32')([input1, input2])
    y_pred = Dense(1, activation='sigmoid', 
            dtype='float32',
            kernel_initializer=weight_init(), 
            bias_initializer=bia_init(), 
            kernel_regularizer=reg(),
    )(x)
    return Model(inputs=(input1, input2), outputs=y_pred, name='sigmoid_model')


@log_model
def build_imagenet_encoder(input_shape, dense_layers, 
            dense_nodes, latent_nodes, 
            dropout_rate, activation, final_activation, pooling=None):
    base_model = pre_trained_model(
        weights='imagenet', include_top=False, pooling=pooling, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    #x = Dense(dense_nodes, activation=activation)(x) # NOTE added to reduce dimensions for later use
    
    #x = Dropout(0.5)(x)

    for _ in range(dense_layers):
        x = Dense(dense_nodes, activation=activation,
                    kernel_initializer=weight_init(),
                    bias_initializer=bia_init(),
                    kernel_regularizer=reg(),
                    )(x)
        x = Dropout(dropout_rate)(x)
    x = Dense(latent_nodes, activation=final_activation, dtype='float32',
            kernel_initializer=weight_init(), 
            bias_initializer=bia_init(), 
            kernel_regularizer=reg(),
            )(x)
    imagenet_encoder_model = Model(base_model.inputs, x, name='imagenet_encoder')
    return imagenet_encoder_model
    #log_summary(imagenet_encoder_model, 'imagenet_encoder.txt')
    #conv_layers.append(imagenet_encoder_model.outputs)


@log_model
def build_custom_encoder(dense_layers, dense_nodes, latent_nodes, activation, final_activation, dropout_rate, 
                    padding='same', pooling='max'):
    model = tf.keras.Sequential(name='custom_encoder', layers=[
        Conv2D(filters=32, kernel_size=(9, 9), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=32, kernel_size=(9, 9), strides=2, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        Conv2D(filters=32, kernel_size=(9, 9), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=32, kernel_size=(9, 9), strides=2, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        Conv2D(filters=64, kernel_size=(7, 7), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        tf.keras.layers.MaxPool2D(pool_size=2),
        Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        Flatten(),
    ])
    for _ in range(dense_layers):
        model.add(Dense(dense_nodes, activation=activation,
                    kernel_initializer=weight_init(),
                    bias_initializer=bia_init(),
                    kernel_regularizer=reg(),
                    ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(latent_nodes, activation=final_activation, 
            dtype='float32',
            kernel_initializer=weight_init(), 
            bias_initializer=bia_init(), 
            kernel_regularizer=reg(),
            ))
    return model

def build_model(should_transfer_learn, latent_nodes, input_shape, *args, **kwargs):
    if should_transfer_learn:
        base_model: tf.keras.Model = build_imagenet_encoder(*args, latent_nodes=latent_nodes, input_shape=input_shape, **kwargs)
    else:
        base_model: tf.keras.Model = build_custom_encoder(*args, latent_nodes=latent_nodes, input_shape=input_shape, **kwargs)
    print(base_model.output_shape)
    print(base_model.output_shape[-1])
    assert base_model.output_shape[-1] == latent_nodes
    head_model: tf.keras.Model = distance_model(base_model.output_shape[-1])
    return combine_models(base_model=base_model, head_model=head_model, name='vanilla'), base_model, head_model
##############################################################################