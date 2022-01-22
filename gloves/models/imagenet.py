from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet50V2 as pre_trained_model
import tensorflow as tf

def build_imagenet_model(freeze):
    imagenet = pre_trained_model(weights='imagenet', include_top=False, input_shape=(224,224,3))
    if freeze:
        imagenet.trainable = False
        # for layer in imagenet.layers:
        #     layer.trainable = False
    x = imagenet.output
    x = Flatten()(x)
    imagenet_model = tf.keras.Model(inputs=imagenet.inputs, outputs=x)
    return imagenet_model

