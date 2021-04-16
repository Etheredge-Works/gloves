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
        downsample: bool, 
        filters: int, 
        kerneal_size: int = 3,
        reg_rate=0.01) -> Tensor:
    y = Conv2D(kernel_size=kerneal_size,
                strides=(1 if not downsample else 2),
                filters=filters,
                kernel_regularizer=reg(reg_rate),
                use_bias=False,
                padding="same")(x)
    y = ReLU()(BatchNormalization()(y))
    y = Conv2D(kernel_size=kerneal_size,
                strides=1,
                filters=filters,
                kernel_regularizer=reg(reg_rate),
                use_bias=False,
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
        ic(stride_width)
        ic(stride_height)
        x = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=reg(reg_rate))(x)
        out = Add()([x,y])
    out = ReLU()(BatchNormalization()(out))
    return out
    

from tensorflow.keras.layers.experimental import preprocessing
def build_custom_encoder(input_shape, dense_layers, dense_nodes, latent_nodes, activation, final_activation, dropout_rate, 
                    padding='same', pooling='max', conv_reg_rate=0.01, dense_reg_rate=0.1):

    input = tf.keras.Input(input_shape)
    x = input
    # semi from resnet https://arxiv.org/pdf/1512.03385.pdf
    x = Conv2D(kernel_size=7,
               strides=2,
               filters=64,
               kernel_regularizer=reg(conv_reg_rate),
               use_bias=False,
               padding="same")(x)
    x = ReLU()(BatchNormalization()(x))
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = block(x, False, 64, reg_rate=conv_reg_rate)
    #x = block(x, False, 64)
    #x = block(x, False, 64)

    x = block(x, True, 128, reg_rate=conv_reg_rate)
    x = block(x, False, 128, reg_rate=conv_reg_rate)
    #x = block(x, False, 128)

    x = block(x, True, 256, reg_rate=conv_reg_rate)
    x = block(x, False, 256, reg_rate=conv_reg_rate)
    #x = block(x, False, 256)

    x = block(x, True, 512, reg_rate=conv_reg_rate)
    x = block(x, False, 512, reg_rate=conv_reg_rate)
    #x = block(x, False, 512)

    x = AvgPool2D(7)(x)
    x = Flatten()(x)

    for _ in range(dense_layers):
        x = Dense(
            dense_nodes, activation=activation,
            kernel_initializer=weight_init(),
            bias_initializer=bia_init(),
            kernel_regularizer=reg(dense_reg_rate),
            )(x)
        #model.add(Dropout(dropout_rate))
    x = Dense(latent_nodes, activation=final_activation, 
            dtype='float32',
            kernel_initializer=weight_init(), 
            bias_initializer=bia_init(), 
            kernel_regularizer=reg(dense_reg_rate),
            )(x)
    model = tf.keras.Model(inputs=input, outputs=x)
    return model

    model_old = tf.keras.Sequential(name='custom_encoder', layers=[
        tf.keras.layers.InputLayer(input_shape),

        # Using stem from https://arxiv.org/pdf/2102.06171.pdf
        Conv2D(filters=16, kernel_size=(3, 3), strides=2, padding='valid', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='linear', use_bias=False,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        BatchNormalization(axis=-1),
        ReLU(),

        #preprocessing.Resizing(height=input_shape[0], width=input_shape[1]),
        #preprocessing.Rescaling(1./255.),
        #Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #Conv2D(input_shape=input_shape, filters=64, kernel_size=(11, 11), strides=2, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        #tf.keras.layers.MaxPool2D(),

        #Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=96, kernel_size=(3, 3), strides=2, padding='valid', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=160, kernel_size=(3, 3), strides=1, padding=padding, activation='linear', use_bias=False,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        BatchNormalization(axis=-1),
        ReLU(),
        #BatchNormalization(),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.AvgPool2D(pool_size=2),

        #tf.keras.layers.MaxPool2D(pool_size=2),
        #Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='valid', activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-2)),
        Conv2D(filters=160, kernel_size=(3, 3), strides=2, padding='valid', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=224, kernel_size=(3, 3), strides=1, padding=padding, activation='linear', use_bias=False,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        BatchNormalization(axis=-1),
        ReLU(),
        #BatchNormalization(),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.AvgPool2D(pool_size=2),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        #Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #Conv2D(filters=256, kernel_size=(5, 5), strides=2, padding='valid', activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-2)),
        
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.MaxPool2D(pool_size=2),
        #Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        Conv2D(filters=224, kernel_size=(3, 3), strides=2, padding='valid', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-2)),
        Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding=padding, activation='linear', use_bias=False,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-1)),
        BatchNormalization(axis=-1),
        ReLU(),
        #BatchNormalization(),
        #tf.keras.layers.MaxPool2D(),

        Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='valid', activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-3)),
        #tf.keras.layers.MaxPool2D(),
        Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding=padding, activation=activation,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-3)),
        Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding=padding, activation='linear', use_bias=False,
                kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg(1e-3)),
        BatchNormalization(axis=-1),
        ReLU(),
        #BatchNormalization(),
        #tf.keras.layers.MaxPool2D(),
        #if pooling: tf.keras.layers.MaxPool2D(pool_size=2),
        #Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        #tf.keras.layers.AvgPool2D(pool_size=7),
        #Conv2D(filters=1024, kernel_size=(14, 14), strides=7, padding=padding, activation=activation,
                #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
        tf.keras.layers.AvgPool2D(pool_size=2),
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