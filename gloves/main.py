#! python
from genericpath import exists
import os
from numpy.lib import ufunclike
from tensorflow.keras.layers import Dense
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import Counter

import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION)
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

from utils import read_decode
from models import build_custom_encoder
os.environ['PYTHONHASHSEED']=str(4)
#import wandb
#wandb.init(project="gloves", config={"hyper":"parameter"})
import mlflow
mlflow.set_experiment("gloves-siamese")
from mlflow import pyfunc
import click
from pathlib import Path

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
# TODO test tensorboard for profiling
import yaml
import tensorflow_addons as tfa
from icecream import ic
#import mlflow


import numpy as np
np.random.seed(4)

tf.random.set_seed(4)
import mlflow.tensorflow
mlflow.tensorflow.autolog(every_n_iter=1)
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)
import pathlib


#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

def log_metric(key, value, step=None):
    mlflow.log_metric(key=key, value=value, step=step)

from siamese.models import Encoder, SiameseModel, create_siamese_model
#from siamese.layers import NormDistanceLayer
from siamese.data import create_dataset, get_labels_from_filenames, create_n_way_dataset

class NormDistanceLayer(tf.keras.layers.Layer):
   def __init__(self, **kwargs):
      super(NormDistanceLayer, self).__init__(**kwargs)

   def call(self, inputs):
      x, y = inputs
      return tf.norm(x-y, axis=-1, keepdims=True)

class AbsDistanceLayer(tf.keras.layers.Layer):
   def __init__(self, **kwargs):
      super(AbsDistanceLayer, self).__init__(**kwargs)

   def call(self, inputs):
      x, y = inputs

      return tf.abs(tf.math.subtract(x,y))

class EuclideanDistanceLayer(tf.keras.layers.Layer):
   def __init__(self, **kwargs):
      super(AbsDistanceLayer, self).__init__(**kwargs)

   def call(self, inputs):
      x, y = inputs

      return tf.sqrt(tf.reduce_sum(tf.math.pow(tf.math.subtract(x,y), 2), axis=-1, keepdims=True))


from copy import deepcopy
def mlflow_log_wrapper(func):
    def inner(*args, **kwargs):
        params = deepcopy(kwargs)
        #for arg in args:
            #arg_name = f'{arg=}'.split('=')[0]
            #params[arg_name] = arg

        mlflow.log_params(params)
        return func(*args, **kwargs)
    return inner

def log_summary(model):
    filename = model.name + ".txt"
    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    mlflow.log_artifact(filename)

import tensorflow as tf
import numpy as np



class NWayCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, 
            encoder: tf.keras.Model, 
            head: tf.keras.Model, 
            nway_ds: tf.data.Dataset, 
            freq: int, 
            prefix_name: str = "",
            batch_size: int = 32,
            *args, **kwargs):
        super(NWayCallback, self).__init__(*args, **kwargs)
        self.encoder = encoder # storing for faster comparisons
        self.head = head 
        # TODO layout structure of nway_ds
        self.freq = freq
        self.prefix_name = prefix_name

        batch, _ = next(iter(nway_ds))
        self.size = len(list(iter(nway_ds)))
        self.nways = len(batch)
        self.batch_size = batch_size

        self.nway_ds = nway_ds.unbatch().batch(batch_size).cache()

    # TODO pull out some of this logic
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            # TODO move to predict from comprehension
            all_encodings = np.reshape(self.encoder.predict(self.nway_ds), (self.size, self.nways, -1))
            ic(all_encodings.shape)
            #all_encodings = [self.encoder.predict_on_batch(item) for item, _ in self.nway_ds]
            #assert(len(all_encodings) > 1)
            predictions = []
            avg_distances = []
            variances = []
            anchors = all_encodings[:, 0, :]
            anchors = [anchors for _ in range(self.nways-1)]
            others = all_encodings[:, 1:, :]
            distances = self.head.predict((anchors, others))
            ic(len(distances))
            ic(len(distances[0]))
            for encodings in all_encodings:
                #assert(len(encodings) > 1)
                anchor = encodings[0]
                anchors = tf.convert_to_tensor([anchor for _ in encodings[1:]])

                # Move expected match to prevent 100% accuracy spike when all distances are equal
                #encodings[1], encodings[2] = encodings[2], encodings[1] # TODO why does this duplicate entry?
                temp = deepcopy(encodings[2])
                encodings[2] = deepcopy(encodings[1])
                encodings[1] = temp

                distances = self.head.predict_on_batch((anchors, tf.convert_to_tensor(encodings[1:])))
                #ic(distances)
                #distances = np.array([self.head((anchor, encoding)) for encoding in encodings[1:]]).flatten()
                #assert(len(distances) > 1)
                # TODO move expected value to last item since all values could be zero causing 100% accuracy due to first item being min
                
                predictions.append(np.argmin(distances))
                avg_distances.append(np.average(distances))
                variances.append(np.var(distances))

            correct_predictions = [prediction == 1 for prediction in predictions]
            score = np.average(correct_predictions)
            logs[f'{self.prefix_name}nway_acc'] = score

            avg_distance = np.average(avg_distances)
            logs[f'{self.prefix_name}nway_avg_dist'] = avg_distance

            avg_variance = np.average(variances)
            logs[f'{self.prefix_name}nway_avg_var'] = avg_variance

from siamese.callbacks import NWayCallback

@click.command()
# File stuff
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--train-extra-dir', default=None, type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--test-extra-dir',  default=None, type=click.Path(exists=True), help='')
@click.option('--model-dir', type=click.Path(exists=False), help='')
@click.option('--model-filename', type=click.STRING, help='')
#@click.option('--encoder-dir', type=click.Path(exists=False), help='')
@click.option('--encoder-model', type=click.Path(exists=False), help='')
###############################################################################
# Image stuff
@click.option("--height", default=224, type=int)
@click.option("--width", default=224, type=int)
@click.option("--depth", default=3, type=int)
@click.option("--mutate-anchor", default=False, type=bool)
@click.option("--mutate-other", default=False, type=bool)
###############################################################################
# Model Stuff
@click.option('--dense-layers', default=2, type=click.INT, help='')
@click.option('--dense-nodes', default=512, type=click.INT, help='')
@click.option("--activation", default='relu', type=str)
@click.option("--latent-nodes", default=128, type=int)
@click.option("--dropout-rate", default=0.5, type=float)
@click.option("--final-activation", default='linear', type=str)
@click.option('--should-transfer-learn', type=click.BOOL, help='')
###############################################################################
# Training Stuff
@click.option('--loss_name', default='contrastive_loss', type=click.STRING, help='')
@click.option('--lr', default=0.00003, type=click.FLOAT, help='')
@click.option('--optimizer', default='adam', type=click.STRING, help='')
@click.option('--epochs', default=1000, type=click.INT, help='')
@click.option('--batch-size', default=32, type=click.INT, help='')
@click.option('--verbose', default=1, type=click.INT, help='')
#@click.option("--validation-ratio", default=0.3, type=float)
@click.option("--eval-freq", default=1, type=int)
@click.option("--reduce-lr-factor", default=0.8, type=float)
@click.option("--reduce-lr-patience", default=10, type=int)
@click.option("--early-stop-patience", default=10, type=int)
@click.option("--mixed-precision", default=False, type=bool)
@click.option("--unfreeze", default=0, type=int)
@click.option("--nway_freq", default=5, type=int)
@click.option("--nways", default=16, type=int)
def main(
        train_dir: str, 
        train_extra_dir: str, 
        test_dir: str, 
        test_extra_dir: str, 
        model_dir: str,
        model_filename: str,
        encoder_model: str,
        ########################
        height: int,
        width: int,
        depth: int,
        mutate_anchor: bool,
        mutate_other: bool,
        ########################
        dense_layers: int, 
        dense_nodes: int, 
        activation: str,
        latent_nodes: int,
        dropout_rate: float,
        final_activation: str,
        should_transfer_learn: bool,
        ########################
        loss_name: str,
        lr: float, 
        optimizer: str,
        epochs: int, 
        verbose: int,
        batch_size: int, 
        eval_freq: int,
        reduce_lr_factor: float,
        reduce_lr_patience: int,
        early_stop_patience: int,
        mixed_precision: bool,
        unfreeze: bool,
        nway_freq,
        nways,
        ):
    print(type(batch_size))
    assert(type(batch_size) == int)

    params = {
        'train_dir': train_dir, 
        'train_extra_dir': train_extra_dir, 
        'test_dir': test_dir, 
        'test_extra_dir': test_extra_dir, 
        'model_dir': model_dir,
        'dense_nodes': dense_nodes,
        'epochs': epochs, 
        'batch_size': batch_size, 
        'lr': lr, 
        'optmizer': optimizer, 
        'should_transfer_learn': should_transfer_learn, 
        'verbose': verbose, 
        'model_filename': model_filename,
    }
    #ic(params)
    mlflow.log_params(params)
    
    if mixed_precision != 0:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    encoder = Encoder(
        conv_activation=activation,
        dense_layers_count=dense_layers,
        dense_nodes=dense_nodes,
        latent_nodes=latent_nodes,
        activation=activation,
        final_activation=final_activation)

    model = SiameseModel(encoder, NormDistanceLayer())
    
    optimizer_switch = {
        'adam': tf.keras.optimizers.Adam
    }
    optimizer = optimizer_switch[optimizer]

    loss_switch = {
        'contrastive_loss': tfa.losses.ContrastiveLoss(),
        'mse': 'mse'
    }
    loss = loss_switch[loss_name]

    #encoder = build_custom_encoder(den)
    model.compile(loss=tfa.losses.ContrastiveLoss(), optimizer=optimizer(lr=lr))
    #model.compile(loss=loss, optimizer=optimizer(lr=lr))

    # TODO extract and pass in
    train_files_tf = tf.io.gfile.glob(str(Path(train_dir)/'**.jpg'))
    train_labels = get_labels_from_filenames(train_files_tf)
    ic(train_labels)
    ic(len(train_labels))
    #ic(tf.size(train_files_tf))
    #ic(train_files_tf)
    #ic(tf.size(train_labels))
    #ic(train_labels)
    assert len(train_files_tf) == len(train_labels)
    assert tf.size(train_files_tf) > 0, "no train files found"
    extra_train_files = tf.io.gfile.glob(str(Path(train_extra_dir)/'**.jpg')) \
        if train_extra_dir \
        else None


    test_files_tf = tf.io.gfile.glob(str(Path(test_dir)/'**.jpg'))
    ic(tf.size(test_files_tf))
    ic(len(test_files_tf))
    test_labels = get_labels_from_filenames(test_files_tf)
    ic(test_labels)
    ic(len(test_labels))
    assert len(test_files_tf) == len(test_labels)
    assert tf.size(test_files_tf) > 0, "no test files found"
    #ic(test_labels)
    extra_test_files = tf.io.gfile.glob(str(Path(test_extra_dir)/'**.jpg')) \
        if test_extra_dir \
        else None

    #all_files_tf  = tf.concat([train_files_tf, test_files_tf])
    ds = create_dataset(
        anchor_items=train_files_tf,
        anchor_labels=train_labels,
        anchor_decode_func=utils.read_decode,
        #other_items=extra_train_files,
        #other_labels=get_labels_from_files_path(extra_train_files),
    )
    #ds = ds.map(lambda anchor, other, label: prepr)
    val_ds = create_dataset(
        anchor_items=test_files_tf,
        anchor_labels=test_labels,
        #anchor_items=train_files_tf,
        #anchor_labels=train_labels,
        anchor_decode_func=utils.read_decode,
        #other_items=train_files_tf, # needed since test set won't have many items
        #other_labels=train_labels
    )

    mlflow.log_param("dataset_size", len(list(ds)))
    mlflow.log_param("validation_dataset_size", len(list(val_ds)))

    #wandb.init(project="siamese")
    #model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=128, epochs=20, callbacks=[WandbCallback()])
    #ic(ds.take(2))
    #for (i, j), k in ds.take(1):
        #ic(i,j,k)
    #anchor, other, label = ds.take(1)[0]
    #ic(anchor, other, label)
    ds = ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.cache() # want a consistent val_ds
    #model.predict(ds.take(3))

    train_hist = model.fit(
        ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_ds,
        validation_freq=eval_freq,
        #steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        shuffle=False,  # TODO dataset should handle shuffling
        callbacks=[
            ReduceLROnPlateau(monitor='loss', factor=reduce_lr_factor, patience=reduce_lr_patience),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patience, verbose=1, restore_best_weights=True),
            #NWayCallback(),
            #WandbCallback(),
            #ModelCheckpoint(filepath=str(pathlib.Path('checkpoints/checkpoint')),save_weights_only=True, monitor='val_accuracy',
                                                #mode='max', save_best_only=True)
        ]
    )
    model.summary()
    encoder.summary()

    history_dict = train_hist.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    mlflow.log_metrics(history_dict)
    #with open(metrics_file_name, 'w') as f:
        #yaml.dump(history_dict, f, default_flow_style=False)
    #print(history_dict)

    #model.save("model", save_format='tf')
    Path(model_dir).mkdir(parents=True)
    model.save(str(Path(str(model_dir))/model_filename))
    #trainable.save(str(Path(str(model_dir))/encoder_model))
    encoder_path = Path(str(encoder_model))
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(encoder_path), save_format='h5')

    #mlflow.log_metrics(history_dict)

    #model, _ = create_model()
    #return model


if __name__ == "__main__":
    main()