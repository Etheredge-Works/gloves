import click
from pathlib import Path
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

    data_ds = data_ds.map(decode)

    label_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    ds = tf.data.Dataset.zip((data_ds, label_ds))
    ds = ds.shuffle(item_count, seed=4, reshuffle_each_iteration=False) # TODO pass seed

    #skip_count = int(label_count*0.2)
    #val_ds = ds.take(skip_count)
    #val_ds = val_ds.batch(batch_size).prefetch(-1).cache()
    #val_ds = val_ds.cache() #TODO why does this give weird error?

    #ds = ds.skip(skip_count)
    #ds = ds.shuffle(item_count, reshuffle_each_iteration=True)
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


def setup_models(encoder_model_path):
    encoder_model_frozen = tf.keras.models.load_model(encoder_model_path)
    for layer in encoder_model_frozen.layers:
        layer.trainable = False

    encoder_model_unfrozen = tf.keras.models.load_model(encoder_model_path)

    imagenet_model_frozen = build_imagenet_model(freeze=True)
    imagenet_model_unfrozen = build_imagenet_model(freeze=False)

    return encoder_model_frozen, encoder_model_unfrozen, imagenet_model_frozen, imagenet_model_unfrozen
    

@click.command()
# File stuff
@click.option('--encoder-model-path', type=click.Path(exists=True), help='')
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
#@click.option('--label', type=str, help='')
@click.option('--batch-size', default=32, type=click.INT, help='')
@click.option('--height', default=224, type=click.INT, help='')
@click.option('--width', default=224, type=click.INT, help='')
@click.option('--epochs', default=4000, type=click.INT, help='')
@click.option('--verbose', default=1, type=click.INT, help='')
@click.option('--model-path', type=click.Path(exists=False), help='')
@click.option('--label-encoder-path', type=click.Path(exists=False), help='')
@click.option('--mixed-precision', default=True, type=click.BOOL, help='')
def main(
        encoder_model_path: str,
        train_dir: str,
        test_dir: str,
        batch_size: int,
        height: int,
        width: int,
        epochs: int,
        verbose: int,
        model_path: Path,
        label_encoder_path: Path,
        mixed_precision: bool,
        ):

    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    ds, label_count, label_encoder = setup_ds(train_dir, batch_size, decode=random_read_decode)
    #with open(label_encoder_path, 'w') as f:
        #joblib.dump(label_encoder, f)
    joblib.dump(label_encoder, label_encoder_path)


    val_ds, val_label_count, _ = setup_ds(test_dir, batch_size, label_encoder, decode=read_decode)

    # Setup models
    encoder_model_frozen, encoder_model_unfrozen, imagenet_model_frozen, imagenet_model_unfrozen = setup_models(encoder_model_path)

    with mlflow.start_run():
        mlflow.log_artifact(label_encoder_path)
        for name, model in [("encoder_frozen", encoder_model_frozen), ("encoder_unfrozen", encoder_model_unfrozen), 
                    ("imagenet_frozen", imagenet_model_frozen), ("imagenet_unfrozen", imagenet_model_unfrozen)]:
            with mlflow.start_run(nested=True):
                mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)
                mlflow.log_artifact(label_encoder_path)

                head = softmax_model(model.output_shape[1:], label_count, dense_nodes=[1024], dropout_rate=0.5)
                classifier = tf.keras.Model(inputs=model.inputs, outputs=head(model.outputs))
                classifier.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), 
                        metrics=['acc', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])

                # TODO binarh

                # TODO will be imbalanced. monitor that
                # TODO resampling methods
                # tf data has some guides on doing that
                # TODO introduce harder cases later
                # TODO test training this to update encoding
                # TODO maybe make two models. one to classify maincoons, one to classify mittens

                classifier.fit(ds, 
                    validation_data=val_ds,
                    validation_freq=1,
                    epochs=epochs, verbose=verbose, callbacks=[
                        ReduceLROnPlateau(monitor='loss', patience=10),
                        EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=True)])

                model_path = Path(str(model_path))/name
                model_path.mkdir(parents=True, exist_ok=True)
                #mlflow.log_artifact(str(model_path))
                mlflow.keras.log_model(classifier, str(name), registered_model_name=f"gloves_{name}")
                classifier.save(str(model_path), save_format='tf')
                print(f"saved: {model_path}")

if __name__ == "__main__":
    main()
