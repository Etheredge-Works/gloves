import click
from pathlib import Path
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.layers import Flatten, Dense
from models import softmax_model
from utils import read_decode, random_read_decode
from siamese.data import get_labels_from_filenames
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as pre_trained_model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as pre_trained_model
import os
from icecream import ic
from sklearn import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import mlflow
mlflow.set_experiment("gloves-classifier")
import mlflow.tensorflow
#from models import build_imagenet_encoder
import numpy as np
import joblib
import dvclive


from tensorflow.keras.callbacks import Callback
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            if type(value) == np.float32:
                value = float(value)
            dvclive.log(metric, value)
        dvclive.next_step()


def setup_ds(train_dir, batch_size, label_encoder=None, decode=random_read_decode):
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
    ds = ds.shuffle(item_count, seed=4, reshuffle_each_iteration=False) # TODO pass seed mess things up?

    #val_ds = val_ds.batch(batch_size).prefetch(-1).cache()
    #val_ds = val_ds.cache() #TODO why does this give weird error?

    ds = ds.batch(batch_size)
    ds = ds.prefetch(-1)

    return ds, label_count, label_encoder
    #return ds, val_ds, label_count


def build_imagenet_model(freeze):
    imagenet = pre_trained_model(weights='imagenet', include_top=False, pooling='max', input_shape=(224,224,3))
    if freeze:
        for layer in imagenet.layers:
            layer.trainable = False
    x = imagenet.output
    x = Flatten()(x)
    imagenet_model = tf.keras.Model(inputs=imagenet.inputs, outputs=x)
    return imagenet_model



@click.command()
# File stuff
@click.option('--encoder-model-path', type=click.Path(exists=True), help='')
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--param-path', type=click.Path(exists=True), help='')
@click.option('--param-parent-key', type=click.STRING, help='')
@click.option('--out-model-path', type=click.Path(exists=False), help='')
@click.option('--out-label-encoder-path', type=click.Path(exists=None), help='')
@click.option('--out-metrics-path', type=click.Path(exists=None), help='')
@click.option('--mixed-precision', default=True, type=click.BOOL, help='')
@click.option('--use-imagenet', default=True, type=click.BOOL, help='')
@click.option('--is-frozen', default=True, type=click.BOOL, help='')
def main(
    encoder_model_path: str,
    train_dir: str,
    test_dir: str,
    param_path,
    param_parent_key,
    out_model_path: Path,
    out_label_encoder_path: Path,
    out_metrics_path,
    mixed_precision: bool,
    use_imagenet,
    is_frozen
):
    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)
    with open(param_path, "r") as f:
        train_kwargs = yaml.load(f)[param_parent_key]
    train(train_dir, test_dir, encoder_model_path, out_model_path, out_metrics_path, out_label_encoder_path,
          use_imagenet=use_imagenet, is_frozen=is_frozen,
          **train_kwargs)

def train(
    train_dir,
    test_dir,
    encoder_model_path,
    out_model_path,
    out_metric_path,
    out_label_encoder_path,
    use_imagenet,
    is_frozen,
    *,
    batch_size,
    epochs,
    verbose,
    **_  # Ignore other kwargs
):

    ds, label_count, label_encoder = setup_ds(train_dir, batch_size, decode=random_read_decode)
    joblib.dump(label_encoder, out_label_encoder_path)

    val_ds, _, _ = setup_ds(test_dir, batch_size, label_encoder, decode=read_decode)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # Setup models
    if use_imagenet:
        model = build_imagenet_model(freeze=is_frozen)
    else:
        model = tf.keras.models.load_model(encoder_model_path)
        if is_frozen:
            for layer in model.layers:
                layer.trainable = False

    dvclive.init(out_metric_path)
    with mlflow.start_run():
        mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)
        mlflow.log_artifact(out_label_encoder_path)

        head = softmax_model(model.output_shape[1:], label_count, dense_nodes=[1024], dropout_rate=0.2)
        classifier = tf.keras.Model(inputs=model.inputs, outputs=head(model.outputs))
        classifier.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['acc', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])

        classifier.fit(ds, 
            validation_data=val_ds,
            validation_freq=1,
            epochs=epochs, verbose=verbose, callbacks=[
                ReduceLROnPlateau(monitor='loss', patience=10),
                MetricsCallback(),
                EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True)])

        classifier.save(out_model_path, save_format='tf')

if __name__ == "__main__":
    main()

# TODO binarh
# TODO will be imbalanced. monitor that
# TODO resampling methods
# tf data has some guides on doing that
# TODO introduce harder cases later
# TODO test training this to update encoding
# TODO maybe make two models. one to classify maincoons, one to classify mittens