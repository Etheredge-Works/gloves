import click
from pathlib import Path
import tensorflow as tf
import custom_model
import os

@click.command()
# File stuff
@click.option('--encoder-model', type=click.Path(exists=True), help='')
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--label', type=str, help='')
@click.option('--model-dir', type=click.Path(exists=False), help='')
def main(
        encoder_model: str,
        train_dir: str,
        label: str,
        model_dir: Path
        ):

    base_model = tf.keras.models.load_model(encoder_model)
    for layer in base_model.layers:
        layer.trainable = False

    # TODO add more layers here
    classifier = custom_model.combine_models(
        base_model=base_model,
        head_model=custom_model.sigmoid_model(base_model.output_shape[-1])
    )

    classifier.compile(optimizer='adam', loss='mse', metrics='acc')

    # TODO will be imbalanced. monitor that
    # TODO resampling methods
    # TODO introduce harder cases later
    # TODO test training this to update encoding
    # TODO maybe make two models. one to classify maincoons, one to classify mittens

    train_dir = Path(train_dir)
    ds = tf.data.Dataset.list_files(
        str(train_dir/r'*.jpg'), shuffle=False)

    all_files_tf = tf.io.gfile.listdir(str(train_dir)) # gets just file name
    labels = tf.strings.regex_replace(all_files_tf, pattern=r'_\d.*', rewrite='')

    def file_decode(file_path):
        # convert the path to a list of path components
        split = tf.strings.split(file_path, os.path.sep)
        tf.debugging.assert_greater(tf.size(split), tf.constant(1), f"Split is wrong.\n")
        file_name = tf.gather(split, tf.size(split) - 1)
        
        label = tf.strings.regex_replace(file_name, r'_\d+\.jpg.*', '')

        one_hot = label == labels

        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img, tf.argmax(tf.cast(one_hot, tf.uint8)) # must cast since is bool

    ds = ds.map(file_decode)
    classifier.fit(ds, batch_size=32, epochs=1, verbose=1)

