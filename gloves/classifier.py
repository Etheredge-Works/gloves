import click
from pathlib import Path
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision
from models import softmax_model, build_imagenet_model
from utils import read_decode, random_read_decode
from siamese.data import get_labels_from_filenames
import os
from icecream import ic
from sklearn import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import mlflow
import mlflow.tensorflow
import numpy as np
import joblib
import dvclive
import wandb


from tensorflow.keras.callbacks import Callback
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            if type(value) == np.float32:
                value = float(value)
            dvclive.log(metric, value)
        dvclive.next_step()


def setup_ds(train_dir, batch_size, label_encoder=None, decode=random_read_decode, reshuffle=True):
    # TODO pass labels?
    train_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(train_dir)/'**.jpg')))
    item_count = int(tf.size(train_files_tf))

    train_labels_string = tf.convert_to_tensor(get_labels_from_filenames(train_files_tf))
    label_count = len(tf.unique(train_labels_string)[0])
    ic(label_count)
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(train_labels_string)
    train_labels = tf.keras.utils.to_categorical(label_encoder.transform(train_labels_string), num_classes=label_count)

    data_ds = tf.data.Dataset.from_tensor_slices(train_files_tf)

    data_ds = data_ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    ds = tf.data.Dataset.zip((data_ds, label_ds))
    ds = ds.shuffle(item_count, seed=4, reshuffle_each_iteration=reshuffle) # TODO pass seed mess things up?

    ds = ds.batch(batch_size)
    ds = ds.prefetch(-1)

    return ds, label_count, label_encoder



@click.command()
# File stuff
@click.option('--mixed_precision', default=False, type=click.BOOL, help='')
@click.option('--encoder_model_path', type=click.Path(exists=True), help='')
@click.option('--train_dir', type=click.Path(exists=True), help='')
@click.option('--test_dir', type=click.Path(exists=True), help='')
@click.option('--out_model_path', type=click.Path(exists=False), help='')
@click.option('--out_label_encoder_path', type=click.Path(exists=None), help='')
@click.option('--out_metrics_path', type=click.Path(exists=None), help='')
@click.option('--use_imagenet', default=True, type=click.BOOL, help='')
@click.option('--is_frozen', default=True, type=click.BOOL, help='')
# passed through (ish)
@click.option('--verbose', default=1, type=click.INT)
@click.option('--batch_size', type=click.INT, help='give power of two')
@click.option('--epochs', type=click.INT)
@click.option('--dropout_rate', type=click.FLOAT)
@click.option('--learning_rate', type=click.FLOAT)
@click.option('--mutate_ds', type=click.BOOL)
@click.option('--lr_monitor_metric', type=click.STRING)
@click.option('--lr_monitor_patience', type=click.INT)
@click.option('--lr_monitor_factor', type=click.FLOAT)
@click.option('--stop_monitor_metric', type=click.STRING)
@click.option('--stop_monitor_patience', type=click.INT)
@click.option('--optimizer', type=click.STRING)
@click.option('--activation', type=click.STRING)
@click.option('--layers', type=click.INT)
@click.option('--nodes', type=click.INT, help='give power of two')
def main(
    mixed_precision: bool,
    **train_kwargs
):

    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    wandb.init(
        project="gloves-classifier",
        config=train_kwargs)
    mlflow.set_experiment("gloves-classifier")
    mlflow.start_run()
    mlflow.log_params(train_kwargs)
    train(**train_kwargs)

def train(
    train_dir,
    test_dir,
    encoder_model_path,
    out_model_path,
    out_metrics_path,
    out_label_encoder_path,
    use_imagenet,
    is_frozen,
    verbose,
    batch_size,
    epochs,
    dropout_rate,
    learning_rate,
    mutate_ds,
    lr_monitor_metric,
    lr_monitor_patience,
    lr_monitor_factor,
    stop_monitor_metric,
    stop_monitor_patience,
    optimizer,
    activation,
    layers,
    nodes,

    #**_  # Ignore other kwargs
):
    nodes = 2**nodes

    ds, label_count, label_encoder = setup_ds(
        train_dir, 
        batch_size, 
        decode=random_read_decode if mutate_ds else read_decode
    )

    if out_label_encoder_path:
        joblib.dump(label_encoder, out_label_encoder_path)

    val_ds, _, _ = setup_ds(test_dir, batch_size, label_encoder, decode=read_decode, reshuffle=False)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # Setup models
    if use_imagenet:
        model = build_imagenet_model(freeze=is_frozen)
    else:
        model = tf.keras.models.load_model(encoder_model_path)
        model.trainable = not is_frozen
        # if is_frozen:
        #     for layer in model.layers:
        #         layer.trainable = False
    if out_metrics_path:
        dvclive.init(out_metrics_path)
    mlflow.tensorflow.autolog(every_n_iter=1)
    if out_label_encoder_path:
        mlflow.log_artifact(out_label_encoder_path)

    head = softmax_model(
        model.output_shape[1:], 
        label_count, 
        dense_nodes=[nodes for _ in range(layers)], 
        dropout_rate=dropout_rate,
        activation=activation,
    )
    classifier = tf.keras.Model(inputs=model.inputs, outputs=head(model.outputs))

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise Exception(f"Unknown optimizer: {optimizer}")

    classifier.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), 
            metrics=[
                'acc', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'), 
                tf.keras.metrics.AUC(name='auc')])

    # TODO add AUC to montior metrics list since imbalanced
    classifier.fit(ds, 
        validation_data=val_ds,
        validation_freq=1,
        epochs=epochs, verbose=verbose, callbacks=[
            ReduceLROnPlateau(monitor=lr_monitor_metric, patience=lr_monitor_patience, factor=lr_monitor_factor),
            MetricsCallback(),
            #EarlyStopping(monitor=stop_monitor_metric, patience=stop_monitor_patience, verbose=1, restore_best_weights=True),
            wandb.keras.WandbCallback(),
        ]
    )

    if out_model_path:
        classifier.save(out_model_path, save_format='tf')
        mlflow.log_artifact(out_model_path)

if __name__ == "__main__":
    main()

# TODO binarh
# TODO will be imbalanced. monitor that
# TODO resampling methods
# tf data has some guides on doing that
# TODO introduce harder cases later
# TODO test training this to update encoding
# TODO maybe make two models. one to classify maincoons, one to classify mittens