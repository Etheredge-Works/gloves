#from src import utils
import utils
utils.limit_gpu_memory_use()

import tensorflow as tf
from tensorflow.keras import Model
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as pre_trained_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as pre_trained_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import numpy as np

import settings

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

def create_model():
    pass


def glovesnet(
        dense_nodes=settings.DENSE_NODES,
        should_transfer_learn=False):
        input1 = tf.keras.Input(input_shape)
        input2 = tf.keras.Input(input_shape)
        if should_transfer_learn:
            base_model = pre_trained_model(weights='imagenet', include_top=False, input_shape=input_shape)
            for layer in base_model.layers:
                layer.trainable = False
            x = base_model.output
            x = Flatten()(x)
            x = Dense(128, activation='relu', dtype='float32',
                      kernel_initializer=weight_init(),
                      bias_initializer=bia_init(),
                      kernel_regularizer=reg()
                      )(x)
            #x = Dropout(0.5)(x)
            encoder_model = Model(base_model.inputs, x)
        else:
            encoder_model = tf.keras.Sequential([
                Conv2D(filters=32, kernel_size=(13, 13), strides=7, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=64, kernel_size=(11, 11), strides=6, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=128, kernel_size=(9, 9), strides=5, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=256, kernel_size=(7, 7), strides=4, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=4),
                Conv2D(filters=512, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=4),
                #Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', activation='relu',
                       #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                # Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=64, kernel_size=(6, 6), strides=3, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', use_bias=False),
                Flatten(),
                Dense(128, activation='relu', kernel_initializer=weight_init(), bias_initializer=bia_init(),
                      kernel_regularizer=reg(),
                      dtype='float32'),
                #Dropout(0.2),  # TODO param
            ])
        print(encoder_model.summary())
        #encoder_input = (None, settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS)

        merge_layer = Lambda(euclidean_distance)([encoder_model(input1), encoder_model(input2)])  # ([self.dense1, self.dense2])
        prediction_layer = Dense(1,
                                activation="sigmoid",
                                kernel_initializer=weight_init(),
                                bias_initializer=bia_init(),
                                kernel_regularizer=reg(),
                                dtype='float32')(merge_layer)
        #self.build(input_shape=(None, (encoder_input, encoder_input)))
        #print(self.summary())
        #model = Model(inputs=[input1, input2], outputs=dense_layer)
        network = Model(inputs=(input1, input2), outputs=prediction_layer)
        print(network.summary())
        return network


class GlovesNet(tf.keras.Model):

    #base_image = utils.simple_decode(tf.io.read_file("data\kitten_mittens.jpg"))  # TODO paramaterize
    #self.base_image = utils.simple_decode(base_file)

    def __init__(self,
                 dense_nodes=settings.DENSE_NODES,
                 should_transfer_learn=False):
        super(GlovesNet, self).__init__()
        if should_transfer_learn:
            base_model = pre_trained_model(weights='imagenet', include_top=False, input_shape=input_shape)
            for layer in base_model.layers:
                layer.trainable = False
            x = base_model.output
            x = Flatten()(x)
            x = Dense(512, activation='relu', dtype='float32',
                      kernel_initializer=weight_init(),
                      bias_initializer=bia_init(),
                      kernel_regularizer=reg()
                      )(x)
            #x = Dropout(0.5)(x)
            self.encoder_model = Model(base_model.inputs, x)
        else:
            self.encoder_model = tf.keras.Sequential([
                Conv2D(filters=32, kernel_size=(13, 13), strides=7, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=64, kernel_size=(11, 11), strides=6, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=128, kernel_size=(9, 9), strides=5, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                Conv2D(filters=256, kernel_size=(7, 7), strides=4, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=4),
                Conv2D(filters=512, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                Conv2D(filters=1024, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                       kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=4),
                #Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='valid', activation='relu',
                       #kernel_initializer=weight_init(), bias_initializer=bia_init(), kernel_regularizer=reg()),
                #tf.keras.layers.MaxPool2D(pool_size=2),
                # Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=64, kernel_size=(6, 6), strides=3, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=128, kernel_size=(4, 4), strides=2, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', use_bias=False),
                # Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu', use_bias=False),
                Flatten(),
                Dense(128, activation='relu', kernel_initializer=weight_init(), bias_initializer=bia_init(),
                      kernel_regularizer=reg(),
                      dtype='float32'),
                #Dropout(0.2),  # TODO param
            ])
        encoder_input = (None, settings.IMG_HEIGHT, settings.IMG_WIDTH, settings.IMG_CHANNELS)
        self.encoder_model.build(encoder_input)
        print(self.encoder_model.summary())
        #base_file = tf.io.read_file("kitten_mittens.jpg")  # TODO paramaterize
        #self.base_image = utils.simple_decode(base_file)

        #self.input1 = Input(input_shape)
        #self.input2 = Input(input_shape)
        #print(type(self.input1))
        #print(type(self.encoder_model))

        #self.dense1 = self.encoder_model(self.input1)
        #self.dense2 = self.encoder_model(self.input2)
        #$self.dense1 = self.encoder_model(self.input1)
        #$self.dense2 = self.encoder_model(self.input2)

        self.merge_layer = Lambda(euclidean_distance)  # ([self.dense1, self.dense2])
        self.prediction_layer = Dense(1,
                                      activation="sigmoid",
                                      kernel_initializer=weight_init(),
                                      bias_initializer=bia_init(),
                                      kernel_regularizer=reg(),
                                      dtype='float32')
        #self.build(input_shape=(None, (encoder_input, encoder_input)))
        #print(self.summary())
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

    #def partial_predict(self, inputs: np.array, *args, **kwargs):
        #model_input = (np.repeat(self.base_image, inputs.shape[0]), inputs)
        #return self.predict(model_input, *args, **kwargs)


#def contrastive_loss(vects):
    #margin = 1
    #square_pred = K.square(y_pred)
    #margin_square = K.square(K.maximum(margin - y_pred, 0))
    #return K.mean(y_true * square_pred + (1 - y_true) * margin_square

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# TODO test set is being paired with itself. pair it with training maybe? maybe not
def create_model(
        train_dir,
        test_dir,
        batch_size=settings.BATCH_SIZE,
        test_ratio: float = settings.TEST_RATIO,
        dense_nodes=settings.DENSE_NODES,
        epochs=10,
        verbose=1) -> (tf.keras.Model, dict):
    #model = GlovesNet(should_transfer_learn=True, dense_nodes=dense_nodes)
    model = glovesnet(dense_nodes, should_transfer_learn=True)
    #print(model.summary())
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), metrics=["accuracy"])

    # TODO extract and pass in
    train_ds, steps_per_epoch, val_ds, test_ds = get_dataset_values(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    batch_size=batch_size)


    #wandb.init(project="siamese")
    #model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=128, epochs=20, callbacks=[WandbCallback()])

    class NWayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 100 == 0:
                #predictions = [self.model.predict_on_batch(n_way_batch) for n_way_batch in test_ds]
                predictions = model.predict(test_ds)
                highest_predictions = [np.argmax(prediction) for prediction in predictions]
                correct_predictions = [highest_prediction == 0 for highest_prediction in highest_predictions]
                score = np.average(correct_predictions)
                print(f"\n\nN-Way Accuracy: {score}\n")

    train_hist = model.fit(
        train_ds,
        epochs=settings.EPOCHS,
        validation_data=val_ds,
        validation_freq=1,
        #steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        shuffle=False,  # TODO dataset should handle shuffling
        callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20),
                   NWayCallback(),
                   ModelCheckpoint(filepath='checkpoints\checkpoint',save_weights_only=True, monitor='val_accuracy',
                                                      mode='max', save_best_only=True)]
    )
    '''
    if epoch == 0:
        print(model.summary())
    if epoch % 10 == 0:
        predictions = [model.predict_on_batch(n_way_batch) for n_way_batch in test_ds]
        #predictions = model.predict(test_ds)
        highest_predictions = [np.argmax(prediction) for prediction in predictions]
        correct_predictions = [highest_prediction == 0 for highest_prediction in highest_predictions]
        score = np.average(correct_predictions)
        print(f"\nN-Way Accuracy: {score}\n")
    '''


    # model.save("model.h5")  # Cannot save subclassed model as it is "unsafe"
    #model.save("model")
    #score = model.evaluate(test_ds)

    return model, train_hist


def save_model(model):
    model.save("model")


def get_model() -> GlovesNet:
    net = glovesnet(should_transfer_learn=True)
    #net.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
    #return tf.keras.models.load_model("model")
    net.load_weights("model_weights")
    return net
    # TODO make model name variable


def run_model():
    model = tf.keras.models.load_model("model")
    train_ds, test_ds, steps_per_epoch, validation_steps = get_dataset_values(
        utils.get_dataset(),
        settings.BATCH_SIZE,
        settings.TEST_RATIO)
    score = model.evaluate(test_ds, steps=validation_steps)
    return score


'''
def gridsearch():
    batch_sizes = [4, 8, 16, 32, 64, 128, 256]
    dense_nodes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    epochs = [1, 2, 5, 8, 10, 15, 20, 25, 30]
    batch_sizes.reverse()
    dense_nodes.reverse()

    print("starting gridsearch")
    best_model = None
    best_score = float("-inf")
    for batch_size in batch_sizes:
        for dense_node in dense_nodes:
            for epoch in epochs:
                model, (loss, score) = create_model(dataset=utils.get_dataset(),
                                                    batch_size=batch_size,
                                                    test_ratio=0.2,
                                                    dense_nodes=dense_node,
                                                    epochs=epoch,
                                                    verbose=0)
                if score > best_score:
                    best_model = model
                    best_model.save("model")
                    print(f"---------------------")
                    best_score = score
                print(f"best: {best_score}")
                print(f"batch_size: {batch_size}")
                print(f"dense_node: {dense_node}")
                print(f"epoch: {epoch}")
'''
