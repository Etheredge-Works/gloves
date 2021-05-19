#! python
from genericpath import exists
import os
from tensorflow.keras.layers import Dense
import dvclive
from dvclive.keras import DvcLiveCallback
import psutil
import mlflow
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION)

class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            float_value = value
            if type(value) == np.float32:
                float_value = float(value)
            dvclive.log(metric, float_value)

        mem = psutil.virtual_memory().used/8/1024/1024/1024
        dvclive.log('memory_use_GB', mem)
        mlflow.log_metric('memory_use_GB', mem)

        dvclive.next_step()


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

from utils import read_decode, random_read_decode
from models import build_custom_encoder, sigmoid_model
os.environ['PYTHONHASHSEED']=str(4)
import mlflow
from mlflow import pyfunc
import click

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# TODO test tensorboard for profiling
import yaml
import tensorflow_addons as tfa
from icecream import ic
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
import mlflow.tensorflow
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)
import pathlib


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

def log_summary(model, dir=None, name=None):
    name = name or model.name
    Path(dir).mkdir(parents=True, exist_ok=True)

    if dir:
        dir = dir + "/"
    filename = f"{dir}{name}.txt"

    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    mlflow.log_artifact(filename)

import tensorflow as tf
import numpy as np

from siamese.callbacks import NWayCallback


def train(
    train_dir: str, 
    train_extra_dir: str, 
    test_dir: str, 
    test_extra_dir: str, 
    out_model_path: str,
    out_encoder_path: str,
    out_metrics_path: str,
    out_summaries_path: str,
    *,  # only take kwargs for hypers
    #checkpoint_dir: str,
    height,
    width,
    depth,
    # hypers
    mutate_anchor,
    mutate_other,
    dense_reg_rate,
    conv_reg_rate,
    #activation,
    latent_nodes,
    final_activation,
    lr,
    optimizer,
    epochs,
    batch_size,
    verbose,
    eval_freq,
    reduce_lr_factor,
    reduce_lr_patience,
    early_stop_patience,
    mixed_precision,
    nway_freq,
    nways,
    use_batch_norm,
    sigmoid_head,
    **_  # Other args in params file to ignore
):
    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    encoder = build_custom_encoder(
        input_shape=(height, width, depth),
        latent_nodes=latent_nodes,
        #activation=activation,
        final_activation=final_activation,
        dense_reg_rate=dense_reg_rate,
        conv_reg_rate=conv_reg_rate,
        use_batch_norm=use_batch_norm,
    )

    input1 = tf.keras.Input(encoder.output_shape[-1])
    input2 = tf.keras.Input(encoder.output_shape[-1])

    if sigmoid_head:
        outputs = Dense(1, activation='sigmoid', dtype='float32')(AbsDistanceLayer(dtype='float32')((input1, input2)))
        head = tf.keras.Model(inputs=(input1, input2), outputs=outputs, name='Distance')
        loss = 'binary_crossentropy'
        nway_comparator = 'max'
        metrics=['acc']
        monitor_metric = 'val_loss'
    else:
        outputs = NormDistanceLayer(dtype='float32')((input1, input2))
        head = tf.keras.Model(inputs=(input1, input2), outputs=outputs, name='NormDistance')
        loss = tfa.losses.ContrastiveLoss()
        nway_comparator = 'min'
        metrics=None
        monitor_metric = 'loss'

    model = create_siamese_model(encoder, head)
    log_summary(encoder, dir=out_summaries_path, name='encoder')
    log_summary(encoder, dir=out_summaries_path, name='head')
    log_summary(encoder, dir=out_summaries_path, name='model')
    
    from tensorflow.keras.optimizers import Adam
    optimizer_switch = {
        'adam': Adam
    }
    optimizer = optimizer_switch[optimizer]

    # TODO extract and pass in
    train_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(train_dir)/'**.jpg')))
    train_labels = tf.convert_to_tensor(get_labels_from_filenames(train_files_tf))
    assert len(train_files_tf) == len(train_labels)
    assert tf.size(train_files_tf) > 0, "no train files found"
    extra_train_files = tf.io.gfile.glob(str(Path(train_extra_dir)/'**.jpg')) \
        if train_extra_dir \
        else None

    test_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(test_dir)/'**.jpg')))
    test_labels = tf.convert_to_tensor(get_labels_from_filenames(test_files_tf))
    assert len(test_files_tf) == len(test_labels)
    assert tf.size(test_files_tf) > 0, "no test files found"
    extra_test_files = tf.io.gfile.glob(str(Path(test_extra_dir)/'**.jpg')) \
        if test_extra_dir \
        else None

    #all_files_tf  = tf.concat([train_files_tf, test_files_tf])
    ds = create_dataset(
        anchor_items=train_files_tf,
        anchor_labels=train_labels,
        anchor_decode_func=random_read_decode if mutate_anchor else read_decode,
        other_decode_func=random_read_decode if mutate_other else read_decode,
        #other_items=extra_train_files,
        #other_labels=get_labels_from_files_path(extra_train_files),
        #repeat=1,
    ).batch(batch_size).prefetch(-1)

    # NOTE can just take from ds here to remove those items from the training set but leave the files available
    #ds = ds.take(1000)
    #ds = ds.map(lambda anchor, other, label: prepr)
    val_ds = create_dataset(
        anchor_items=test_files_tf,
        anchor_labels=test_labels,
        #anchor_items=train_files_tf,
        #anchor_labels=train_labels,
        anchor_decode_func=read_decode,
        #other_items=train_files_tf, # needed since test set won't have many items
        #other_labels=train_labels
    ).batch(batch_size).cache().prefetch(-1)
    # TODO should val_ds be cached? or should it change?

    test_nway_ds = create_n_way_dataset(
        items=test_files_tf, 
        labels=test_labels,
        ratio=1.0, 
        anchor_decode_func=read_decode, 
        n_way_count=nways)

    nway_ds = create_n_way_dataset(
        items=train_files_tf, 
        labels=train_labels,
        ratio=0.05, 
        anchor_decode_func=read_decode, 
        n_way_count=nways)
    
    mlflow.log_param("dataset_size", len(list(ds))*batch_size)
    mlflow.log_param("validation_dataset_size", len(list(val_ds))*batch_size)

    #wandb.init(project="siamese")
    # TODO how can I use preprocessing layers? Dataset requires images to be the same size for batching...
    nway_callbacks = [] if sigmoid_head else [
        NWayCallback(encoder=encoder, head=head, nway_ds=nway_ds, freq=nway_freq, comparator=nway_comparator, prefix_name="train_"),
        NWayCallback(encoder=encoder, head=head, nway_ds=test_nway_ds, freq=nway_freq, comparator=nway_comparator, prefix_name="test_")]
    callbacks=[
        ReduceLROnPlateau(monitor=monitor_metric, factor=reduce_lr_factor, patience=reduce_lr_patience),
        *nway_callbacks,
        EarlyStopping(monitor=monitor_metric, min_delta=0, patience=early_stop_patience, verbose=1, restore_best_weights=True),
        MetricsCallback(),
    ]

    # TODO remove model from here and have it submit a post request to locally running rest api
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
    train_hist = model.fit(
        ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_ds,
        validation_freq=eval_freq,
        #steps_per_epoch=steps_per_epoch,
        #verbose=verbose,
        shuffle=False,  # TODO dataset should handle shuffling
        callbacks=callbacks
    )
    # TODO remove nway from sigmoid training

    history_dict = train_hist.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    #with open(metrics_file_name, 'w') as f:
        #yaml.dump(history_dict, f, default_flow_style=False)
    #print(history_dict)

    model.save(out_model_path, save_format='tf')
    mlflow.log_artifact(out_model_path)
    encoder.save(out_encoder_path, save_format='tf')
    mlflow.log_artifact(out_metrics_path)


@click.command()
# File stuff
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--train-extra-dir', default=None, type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--test-extra-dir',  default=None, type=click.Path(exists=True), help='')
@click.option('--param-path',  default=None, type=click.Path(exists=True), help='')
@click.option('--param-parent-key',  default='train', type=click.STRING, help='')
@click.option('--out-model-path', type=click.Path(exists=None), help='')
@click.option('--out-encoder-path', type=click.Path(exists=None), help='')
@click.option('--out-metrics-path', type=click.Path(exists=None), help='')
@click.option('--out-summaries-path', type=click.Path(exists=None), help='')
@click.option("--sigmoid-head", default=False, type=bool)
@mlflow_log_wrapper
def main(
        train_dir: str, 
        train_extra_dir: str, 
        test_dir: str, 
        test_extra_dir: str, 
        param_path: str, 
        param_parent_key: str,
        out_model_path: str,
        out_encoder_path: str,
        out_metrics_path: str,
        out_summaries_path: str,
        sigmoid_head: bool,
):
    if sigmoid_head:
        mlflow.set_experiment("siamese-distance")
    else:
        mlflow.set_experiment("siamese-sigmoid")
    mlflow.tensorflow.autolog(every_n_iter=1)

    dvclive.init(out_metrics_path, summary=True, html=True)
    with open(param_path, "r") as f:
        train_kwargs = yaml.load(f)[param_parent_key]

    train(
        train_dir=train_dir,
        train_extra_dir=train_extra_dir,
        test_dir=test_dir,
        test_extra_dir=test_extra_dir,
        sigmoid_head=sigmoid_head,
        out_model_path=out_model_path,
        out_encoder_path=out_encoder_path,
        out_metrics_path=out_metrics_path,
        out_summaries_path=out_summaries_path,
        **train_kwargs)


if __name__ == "__main__":
    limit_gpu_memory_use()
    main()
