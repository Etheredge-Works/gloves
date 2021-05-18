import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as pre_trained_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda, BatchNormalization, ReLU, Add, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np
#from wandb.keras import WandbCallback
import mlflow
from tensorflow.python.keras.layers.pooling import AvgPool2D, MaxPool2D
mlflow.tensorflow.autolog(every_n_iter=1)
from icecream import ic

from utils import settings
import pathlib

from utils import euclidean_distance, input_shape, get_dataset_values, limit_gpu_memory_use
limit_gpu_memory_use()


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


# TODO why were these making the model default all it's predictions to the same thing?
def weight_init(mean=0, stddev=0.01):
    return 'glorot_uniform' #TODO investigate
    values = np.rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
    return tf.random_normal_initializer(mean=mean, stddev=stddev)


def bia_init(mean=0.5, stddev=0.01):
    return 'zeros'
    return tf.random_normal_initializer(mean=mean, stddev=stddev)

def reg(rate=2e-4):
    #return None
    return l2(rate)


#def contrastive_loss(vects):
    #margin = 1
    #square_pred = K.square(y_pred)
    #margin_square = K.square(K.maximum(margin - y_pred, 0))
    #return K.mean(y_true * square_pred + (1 - y_true) * margin_square

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


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
#@log_model
def combine_models(base_model, head_model, name="no_name"):
    #print(base_model.input_shape)
    #print(base_model.output_shape)
    #print(head_model.input_shape)
    #print(head_model.output_shape)
    encoder_inputs = tf.keras.Input(base_model.input_shape[1:]), tf.keras.Input(base_model.input_shape[1:])
    encoder_outputs = head_model([base_model(encoder_inputs[0]), base_model(encoder_inputs[1])])
    return Model(name=name, inputs=encoder_inputs, outputs=encoder_outputs)


#@log_model
def distance_model(input_shape):
    input1 = tf.keras.Input(input_shape)
    input2 = tf.keras.Input(input_shape)
    y_pred = DistanceLayer(dtype='float32')([input1, input2])
    return Model(inputs=(input1, input2), outputs=y_pred, name='distance_model')


#@log_model
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

#@log_model
def softmax_model(input_shape, label_count, 
            dense_nodes: list = [],
            activation='relu',
            dropout_rate=0.0):
    input1 = tf.keras.Input(input_shape)
    # TODO add dense here?
    x = input1
    for node_count in dense_nodes:
        x = Dense(node_count, activation=activation)(x)
        x = Dropout(dropout_rate)(x)
    y_pred = Dense(label_count, activation='softmax', 
            dtype='float32',
            #kernel_initializer=weight_init(), 
            #bias_initializer=bia_init(), 
            #kernel_regularizer=reg(),
    )(x)
    return Model(inputs=input1, outputs=y_pred, name='softmax_model')

#@log_model
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


from tensorflow import Tensor
# https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
def block(
        x: Tensor, 
        filters: int, 
        downsample: bool = False, 
        kerneal_size: int = 3,
        reg_rate=0.01,
        use_batch_norm=True) -> Tensor:
    y = Conv2D(kernel_size=kerneal_size,
                strides=(1 if not downsample else 2),
                filters=filters,
                kernel_regularizer=reg(reg_rate),
                use_bias=not use_batch_norm, # Don't use bias if using batch norm
                padding="same")(x)
    if use_batch_norm: 
        # TODO test layer norm
        y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(kernel_size=kerneal_size,
                strides=1,
                filters=filters,
                kernel_regularizer=reg(reg_rate),
                use_bias=not use_batch_norm,
                padding="same")(y)

    try:
        out = Add()([x,y])
    except ValueError:
        # Sizes don't align
        # https://github.com/raghakot/keras-resnet/blob/master/resnet.py
        input_shape = K.int_shape(x)
        residual_shape = K.int_shape(y)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        x = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=reg(reg_rate))(x)
        out = Add()([x,y])
    if use_batch_norm: 
        out = BatchNormalization()(out)
    out = ReLU()(out)
    return out
    

from tensorflow.keras.layers.experimental import preprocessing
def build_custom_encoder(
    input_shape, 
    #dense_layers, 
    #dense_nodes, 
    latent_nodes, 
    #activation, 
    final_activation, 
    #dropout_rate, 
    #padding='same', 
    #pooling='max', 
    conv_reg_rate=0.01, 
    dense_reg_rate=0.1, 
    use_batch_norm=True,
    latent_dense=False):
    # TODO pass activation as none and assign. str value so constructs in block.
    # TODO maybe linear final layer

    input = tf.keras.Input(input_shape)
    x = input

    # semi from resnet https://arxiv.org/pdf/1512.03385.pdf
    x = Conv2D(kernel_size=7,
               strides=2,
               filters=32,
               kernel_regularizer=reg(conv_reg_rate),
               use_bias=False,
               padding="same")(x)
    x = ReLU()(BatchNormalization()(x))
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = block(x, 32, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)
    x = block(x, 32, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)

    x = block(x, 64, downsample=True, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)
    x = block(x, 64, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)

    x = block(x, 128, downsample=True, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)
    x = block(x, 128, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)

    x = block(x, 256, downsample=True, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)

    # TODO why didn't this work well? lack of sigmoid? not expressive enough?
    #x = block(x, latent_nodes, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)
    if not latent_dense:
        x = block(x, latent_nodes, reg_rate=conv_reg_rate, use_batch_norm=use_batch_norm)

    # TODO not using sigmoid here?...
    x = GlobalAveragePooling2D(dtype='float32')(x)
    x = Flatten(dtype='float32')(x)

    # https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    # They use sigmoid here
    #for _ in range(dense_layers):
        #x = Dense(
            #dense_nodes, activation=activation,
            #kernel_initializer=weight_init(),
            #bias_initializer=bia_init(),
            #kernel_regularizer=reg(dense_reg_rate),
            #)(x)
        #x = Dropout(dropout_rate)(x)
    if latent_dense:
        x = Dense(latent_nodes, activation=final_activation, 
                dtype='float32',
                kernel_initializer=weight_init(), 
                bias_initializer=bia_init(), 
                kernel_regularizer=reg(dense_reg_rate),
                )(x)
    model = tf.keras.Model(inputs=input, outputs=x)
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
