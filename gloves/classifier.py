import click
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.layers import Flatten, Dense
from models import softmax_model
from utils import read_decode
from siamese.data import get_labels_from_filenames
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as pre_trained_model
import os
from icecream import ic
from sklearn import preprocessing
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import mlflow
mlflow.set_experiment("gloves-classifier")
import mlflow.tensorflow
mlflow.tensorflow.autolog(every_n_iter=1)
from models import build_imagenet_encoder
import numpy as np

@click.command()
# File stuff
@click.option('--encoder-model-path', type=click.Path(exists=True), help='')
@click.option('--train-dir', type=click.Path(exists=True), help='')
#@click.option('--label', type=str, help='')
@click.option('--batch-size', default=32, type=click.INT, help='')
@click.option('--height', default=224, type=click.INT, help='')
@click.option('--width', default=224, type=click.INT, help='')
@click.option('--epochs', default=10, type=click.INT, help='')
@click.option('--verbose', default=1, type=click.INT, help='')
@click.option('--model-path', type=click.Path(exists=False), help='')
@click.option('--mixed-precision', default=True, type=click.BOOL, help='')
def main(
        encoder_model_path: str,
        train_dir: str,
        batch_size: int,
        height: int,
        width: int,
        epochs: int,
        verbose: int,
        model_path: Path,
        mixed_precision: bool,
        ):

    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    # TODO pass labels?
    train_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(train_dir)/'**.jpg')))

    train_labels_string = tf.convert_to_tensor(get_labels_from_filenames(train_files_tf))
    label_count = len(tf.unique(train_labels_string)[0])
    ic(label_count)
    le = preprocessing.LabelEncoder()
    le.fit(train_labels_string)
    #train_labels = le.transform(train_labels_string)
    train_labels = tf.keras.utils.to_categorical(le.transform(train_labels_string), num_classes=label_count)


    # Setup datasets
    data_ds = tf.data.Dataset.from_tensor_slices(train_files_tf)
    data_ds = data_ds.map(read_decode)

    label_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    ds = tf.data.Dataset.zip((data_ds, label_ds))
    val_ds = ds.take(int(label_count*0.2))

    ds = ds.skip(int(label_count*0.2))
    #ds = ds.cache()
    ds = ds.shuffle(buffer_size=len(train_labels), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(-1)

    val_ds = val_ds.batch(batch_size)
    
    # Seup models
    base_model = tf.keras.models.load_model(encoder_model_path)
    for layer in base_model.layers:
        layer.trainable = False

    # TODO add more layers here
    #classifier = tf.keras.Model(inputs=input1, outputs=head(base_model(input1)))
    #classifier = build_imagenet_encoder((height,width,3), 2, 
    imagenet = pre_trained_model(weights='imagenet', include_top=False, pooling='max', input_shape=(224,224,3))
    for layer in imagenet.layers:
        layer.trainable = False
    x = imagenet.output
    x = Flatten()(x)
    #base_model = tf.keras.Model(inputs=imagenet.inputs, outputs=x)

    '''
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(label_count, dtype='float32', activation='softmax')(x)

    #head = softmax_model((np.prod(imagenet.output_shape[1:]),), label_count, dense_nodes=[512, 512])
    classifier.summary()
    '''
    head = softmax_model(base_model.output_shape[1:], label_count, dense_nodes=[1024], dropout_rate=0.0)
    classifier = tf.keras.Model(inputs=base_model.inputs, outputs=head(base_model.outputs))


    classifier.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics='acc')
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
            ReduceLROnPlateau(monitor='loss', factor=0.8, patience=4),
            EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)])

    model_path = Path(str(model_path))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(str(model_path), save_format='tf')
    print(f"saved: {model_path}")

if __name__ == "__main__":
    main()
