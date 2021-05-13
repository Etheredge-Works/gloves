import click
import tensorflow as tf
import tensorflow.keras.backend as K
#from tensorflow.python.keras.layers.core import Dropout
from classifier import setup_ds
from pathlib import Path
import joblib
import mlflow
mlflow.set_experiment("hydra-digits")
from models import block
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import AvgPool2D, Flatten, Dense, Softmax, ReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from icecream import ic
import numpy as np


from contextlib import redirect_stdout
def log_model(model, name):
   file_path = f"/tmp/{name}.txt"
   with open(file_path, 'w') as f:
      with redirect_stdout(f):
         model.summary()
   mlflow.log_artifact(file_path)


def head(x, n_classes, dropout_ratio=0.5, dense_nodes=[128, 128], name=None):
   input_ = Input(K.int_shape(x)[1:])
   y = input_

   # TODO learn about droupout across time
   while(K.int_shape(y)[1] > 2):
      y = block(y, 64, downsample=True, use_batch_norm=True)
   y = AvgPool2D(K.int_shape(y)[1])(y)
   y = Flatten()(y)
   for dense_node_count in dense_nodes:
      y = Dense(dense_node_count)(y)
      y = Dropout(dropout_ratio)(y)
      y = ReLU()(y)

   y = Dense(n_classes, dtype='float32')(y)
   y = Softmax(dtype='float32')(y)
   model = Model(input_, y, name=name)
   log_model(model, name)
   return model(x)

# TODO decay the impact of the heads
# TODO use multiple losses
def hydra_model(input_shape, n_classes, use_batch_norm=True):
   input_ = Input(input_shape)
   heads = []
   x = input_

   # TODO reccurent vs regular?
   x = block(x, 64, downsample=True, use_batch_norm=use_batch_norm)
   x = block(x, 64, downsample=False, use_batch_norm=use_batch_norm)
   #heads.append(head(x, n_classes, name="head_0"))
   x = block(x, 64, downsample=False, use_batch_norm=use_batch_norm)
   x = block(x, 64, downsample=False, use_batch_norm=use_batch_norm)
   #heads.append(head(x, n_classes, name="head_1"))

   x = block(x, 128, downsample=True, use_batch_norm=use_batch_norm)
   x = block(x, 128, downsample=False, use_batch_norm=use_batch_norm)
   #heads.append(head(x, n_classes, name="head_2"))
   x = block(x, 128, downsample=False, use_batch_norm=use_batch_norm)
   x = block(x, 128, downsample=False, use_batch_norm=use_batch_norm)
   #heads.append(head(x, n_classes, name="head_3"))

   #x = block(x, 256, downsample=True, use_batch_norm=use_batch_norm)
   #x = block(x, 256, downsample=False, use_batch_norm=use_batch_norm)
   #heads.append(head(x, n_classes, name="head_4"))
   #x = block(x, 256, downsample=False, use_batch_norm=use_batch_norm)
   #x = block(x, 256, downsample=False, use_batch_norm=use_batch_norm)
   heads.append(head(x, n_classes, name="final_head"))

   model = Model(input_, heads)
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
   return model, len(heads)

# TODO test avg vs max
# TODO max may isolate parts of the network to that functinoality
# TODO avg may make the information more spread out

# TODO does the early heads get forced to suck due to weight of later heads?
# TODO does the early heads act as a regularizer?
# TODO can I make the earlier heads not propogate base their layers? is this valuaable? I don't think so. It's just like taking on to the finished encoders layers
#      and it's similar to online learning as the head is learning along side the encoder, but it does not enforce itself on the encoder? 
#      could this resutl in a better head?

@click.command()
# File stuff
#@click.option('--train-dir', type=click.Path(exists=True), help='')
#@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--batch-size', default=128, type=click.INT, help='')
@click.option('--epochs', default=1000, type=click.INT, help='')
@click.option('--verbose', default=1, type=click.INT, help='')
#@click.option('--model-path', type=click.Path(exists=False), help='')
#@click.option('--label-encoder-path', type=click.Path(exists=False), help='')
@click.option('--mixed-precision', default=True, type=click.BOOL, help='')
def main(
        #train_dir: str,
        #test_dir: str,
        batch_size: int,
        epochs: int,
        verbose: int,
        #model_path: Path,
        #label_encoder_path: Path,
        mixed_precision: bool,
        ):
   mlflow.tensorflow.autolog(every_n_iter=1)
      
   if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)


   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
   #x_train = np.expand_dims(x_train, axis=-1)
   #x_test = np.expand_dims(x_test, axis=-1)
   ic(x_train.shape)

   ic(y_test)
   model, heads_size = hydra_model(x_train.shape[1:], n_classes=len(np.unique(y_train)))

   hydra_y_train = [y_train for _ in range(heads_size)] if heads_size > 1 else y_train
   hydra_y_test = [y_test for _ in range(heads_size)] if heads_size > 1 else y_test
   model.summary()
   #trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
   #mlflow.log_param("trainable_params", trainable_count)
   #non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
   #mlflow.log_param("non_trainable_params", non_trainable_count)


   model.fit(x_train, hydra_y_train, 
         batch_size=batch_size, 
         epochs=epochs, 
         validation_data=(x_test, hydra_y_test), 
         verbose=verbose,
         callbacks=[
            ReduceLROnPlateau(monitor='loss', patience=10),
            EarlyStopping(monitor='val_loss', patience=60, verbose=1, restore_best_weights=True),
         ])

   #with mlflow.start_run():
      #mlflow.log_artifact(label_encoder_path)
      #model = hydra_model(10)

if __name__ == '__main__':
   main()
