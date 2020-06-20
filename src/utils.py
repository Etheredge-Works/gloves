import base64
import os
import pathlib
import settings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import backend as K
import pathlib
import os
import numpy as np
import re

#DATA_DIR = pathlib.Path('data')
#TF_DATA_DIR = tf.constant(str(DATA_DIR))
#CLASS_NAMES = np.array(list(set(
    #[re.sub(r'_\d.*', '', item.name)
     #for item in DATA_DIR.glob('*.jpg')])))
input_shape = (settings.IMG_WIDTH, settings.IMG_HEIGHT, 3)

def get_dataset(dir="images", url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"):
    #raw_data_dir = pathlib.Path(get_file(dir, url, untar=True))
    # remove dumb mat files
    # TODO handle mat files better
    #mat_files = raw_data_dir.glob("*mat")
    #for file in mat_files:
        #os.remove(file)
    #return raw_data_dir
    return pathlib.Path("data/images")

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

ALL = [
    'Abyssinian',
    'yorkshire_terrier',
    'american_bulldog',
    'american_pit_bull_terrier'
]

# TODO could change to glob of jpgs
#ALL_FILES = tf.io.gfile.listdir(str(DATA_DIR))
#print(f"all_files: {ALL_FILES}")
def get_pair(data_dir, all_files, labels, anchor_file_path,
             label=None):
    if label is None:
        label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)
    #anchor_label = get_label(anchor_file_path)
    # TODO tweak random var
    # TODO might have two of the anchor, but oh well
    # TODO make more efecient

    #tf.print(anchor_file_path)
    split_path = tf.strings.split(anchor_file_path, sep=os.path.sep)
    #tf.print(anchor_file_path)
    tf.debugging.assert_greater(tf.size(split_path), tf.constant(1), f"Split is wrong.\n")
    #tf.print(tf.size(split_path))
    #file_name = tf.gather(split_path, 2)
    file_name = tf.gather(split_path, tf.size(split_path) - 1)
    #file_name = 'British_Shorthair_174.jpg'
    anchor_label = get_label(file_name)


    #data_dir = tf.strings.join(tf.gather(split_path, range(tf.size(split_path)-1)), separator=os.path.sep)
    #data_dir = tf.strings.regex_replace(anchor_file_path, fr'{os.path.sep}[^[\{os.path.sep}]_\d+\.jpg$', '') # TODO regex issues with pathsep
    #tf.print(data_dir)

    #labels = tf.strings.regex_replace(ALL_FILES, pattern=r'_\d.*', rewrite='')
    #tf.print(anchor_label)
    #tf.print(labels)
    pos_mask = anchor_label == labels

    #label = tf.cast(tf.math.round(tf.random.uniform([], maxval=1, dtype=tf.float32)), dtype=tf.int32)
    #tf.print(label)

    pos_label_func = lambda: tf.math.logical_xor(pos_mask, all_files == file_name)  # XOR prevents anchor file being used
    neg_label_func = lambda: tf.math.logical_not(pos_mask)
    mask = tf.cond(label == tf.constant(1), pos_label_func, neg_label_func)
    values = tf.boolean_mask(all_files, mask)
    '''
    # TODO implement a way to grab easy, medium, hard pairs (e.g. boxer-cat, boxer-dog, boxer-pug or with losses)
    # TODO Need a way to monitor losses and let it inform seletion weights
    if random_value < -1.33333:
        # get easy
        pass

    elif random_value < -1.66666666666:
        # get semi_hard
        pass
    else:
        # get_hard
        pass
    '''

    tf.debugging.assert_greater(tf.size(values), tf.constant(0), f"Values are empty.\n")
    idx = tf.random.uniform([], 0, tf.size(values), dtype=tf.int32)
    value = tf.gather(values, idx)
    path = tf.strings.join([data_dir, value], os.path.sep)
    sq_path = tf.squeeze(path)
    return anchor_file_path, sq_path, label


def get_label(file_name):
    #assert (len(str(file_path)) > 0)
    #st.write(file_path)
    # Get file name
    #file_path = str(file_path)
    #tf.print(file_path, "file_path") file_name = tf.strings.split(file_path, sep=os.path.sep)[-1]
    #tf.print(file_name, "file_name")
    #label = tf.strings.regex_replace(file_path, f'.*{os.path.sep}', '')

    # Strip down to classname
    label = tf.strings.regex_replace(file_name, r'_\d+\.jpg.*', '')
    # TODO can use digit to make sure anchor and positive are different
    #assert (len(str(label)) > 0)
    #st.write(f"label: {label}")
    return label

#def get_encoded_label(file_path):
    #return get_label(file_path) == CLASS_NAMES


def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: tf.image.random_crop(x, size=(settings.IMG_HEIGHT, settings.IMG_WIDTH, 3)))


def simple_decode(img):
    img = tf.image.decode_jpeg(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)  # NOTE: Must do this before other operations or can mangle img
    # TODO the above was causing issues? I don't understand anymore...
    # TODO I know it should cause issues with mixed precision but why was it handicaping performance?
    img = tf.image.resize(img, [settings.IMG_WIDTH, settings.IMG_HEIGHT])
    img = preprocess_input(img)  # NOTE: This does A TON for accuracy
    #img = tf.image.convert_image_dtype(img, tf.float32)  #TODO remove this if preproces sis used
    return img


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor

    # img = tf.image.decode_jpeg(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [settings.IMG_WIDTH, settings.IMG_HEIGHT])
    # img = tf.image.decode_image(img)
    # img = tf.image.decode_jpeg(img)
    # img = tf.image.decode_image(img, channels=0)
    # img = tf.image.decode_jpeg(img, channels=0)
    # img = tf.image.resize(img, [IMG_WIDTH*2, IMG_HEIGHT*2])
    # img = zoom(img)

    # st.write(img)

    # img = tf.image.convert_image_dtype(img, tf.float32)

    #NUM_BOXES = 3
    #boxes = tf.random.uniform(shape=(NUM_BOXES, 3))
    #box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=settings.BATCH_SIZE, dtype=tf.int32)
    #img = tf.image.crop_and_resize(img, boxes, box_indices, (settings.IMG_HEIGHT, settings.IMG_WIDTH))
    # st.write(img)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # resize the image to the desired size.
    # img = tf.image.random_crop(img, [IMG_WIDTH, IMG_HEIGHT, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # TODO test if this is corrupting jpg
    img = tf.image.random_hue(img, 0.02)
    #img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.02)
    img = tf.image.random_contrast(img, 0.02, 0.05)

    # img = tf.image.convert_image_dtype(img, tf.float16)

    # img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    img = preprocess_input(img)  # This handles float conversion
    # img = tf.image.resize(img, [settings.IMG_WIDTH, settings.IMG_HEIGHT])
    #img = tf.image.convert_image_dtype(img, tf.float32)  #TODO remove this if preproces sis used

    # st.write(img)
    return img


def read_images(data_dir, anchor_decode_func, other_decode_func):
    all_files_tf = tf.io.gfile.listdir(data_dir)
    # TODO will all labels be in testing set?
    labels = tf.strings.regex_replace(all_files_tf, pattern=r'_\d.*', rewrite='')
    data_dir_tf = tf.constant(str(data_dir))

    def foo(file_path):
        anchor_file_path, other_file_path, label = get_pair(str(data_dir), all_files_tf, labels, file_path)
        # TODO may not need to check for different anchor/positive since they'll get morephed differently...
        anchor_file = tf.io.read_file(anchor_file_path)
        other_file = tf.io.read_file(other_file_path)
        #negative_file = tf.io.read_file(negative_file_path)

        anc_img = anchor_decode_func(anchor_file) # TODO maybe do no encoding to anc
        other_img = other_decode_func(other_file)
        return (anc_img, other_img), label

    return foo


def n_way_read(data_dir, decode_func, n):
    all_files_tf = tf.io.gfile.listdir(data_dir)
    labels = tf.strings.regex_replace(all_files_tf, pattern=r'_\d.*', rewrite='')
    count = 0

    def foo(file_name):
        anchors, others = [], []
        labels_list = []  # Some keras interfaces (like predict) expect a label even when not used, so we'll get them too
        pos_anchor, pos_other, label = get_pair(str(data_dir), all_files_tf, labels, file_name, label=1)
        anchors.append(decode_func(tf.io.read_file(pos_anchor)))
        others.append(decode_func(tf.io.read_file(pos_other)))
        labels_list.append(label)
        # TODO will repeat some neagtives, but that's fine
        for _ in range(n-1):
            anchor, other, label = get_pair(str(data_dir), all_files_tf, labels, file_name, label=0)
            anchors.append(decode_func(tf.io.read_file(anchor)))
            others.append(decode_func(tf.io.read_file(other)))
            labels_list.append(label)


        #neg_inputs = [(decode_func(tf.io.read_file(anchor)), decode_func(tf.io.read_file(other))) for anchor, other in neg_inputs]

        return (tf.convert_to_tensor(anchors), tf.convert_to_tensor(others)), tf.convert_to_tensor(labels_list)
    return foo


def prepare_for_training(ds, cache=False, shuffle=True, batch_size=1, shuffle_buffer_size=None, repeat=None):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if shuffle:
        if shuffle_buffer_size is None:
            #shuffle_buffer_size = batch_size  # not a great choice but oh well
            raise ValueError
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    ds = ds.repeat(repeat)

    if batch_size:
        ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=settings.AUTOTUNE)

    return ds


def euclidean_distance(vects):
    x, y = vects
    return K.abs(x-y)  # TODO test other distance metrics
    # TODO seems abs is performing better than euclidean...
    #sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    #return K.maximum(sum_square, K.epsilon())
    #sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    #return K.maximum(sum_square, K.epsilon())


#def n_way_verification(model: tf.keras.Model, data_dir, file_name, n=32):
    #all_files_tf = tf.io.gfile.listdir(str(data_dir))
    #labels = tf.strings.regex_replace(all_files_tf, pattern=r'_\d.*', rewrite='')

    #*pos_inputs, _ = get_pair(str(data_dir), all_files_tf, labels, file_name, label=1)
    ## TODO will repeat some neagtives, but that's fine
    #*neg_inputs, _ = [get_pair(str(data_dir), all_files, labels, file_name, label=0)[0] for _ in range(n-1)]
    #batch = [pos_inputs, *neg_inputs]
    #predictions = model.predict_on_batch(batch)
    #return np.argmax(predictions) == 0


def create_n_way_dataset(data_directory_name, batch_size, anchor_decode_func,
                         n_way_count):
    data_directory = pathlib.Path(data_directory_name)
    all_files = list(data_directory.glob('*.jpg'))
    file_count = len(all_files)
    step_per_epoch = file_count * n_way_count
    ds = tf.data.Dataset.list_files(str(data_directory / '*.jpg'))

    #ds = ds.cache()
    ds_labeled = ds.map(n_way_read(str(data_directory), anchor_decode_func, n=n_way_count),
                        num_parallel_calls=settings.AUTOTUNE)
    ds_prepared = prepare_for_training(ds_labeled, cache=False, shuffle=False, batch_size=None,
                                       shuffle_buffer_size=file_count, repeat=1)
    return ds_prepared



def create_dataset(data_directory_name, batch_size, anchor_decode_func,
                   other_decode_func=None,
                   shuffle=False,
                   repeat=None):
    if other_decode_func is None:
        other_decode_func = anchor_decode_func

    data_directory = pathlib.Path(data_directory_name)
    all_files = list(data_directory.glob('*.jpg'))
    file_count = len(all_files)

    step_per_epoch = file_count // batch_size

    ds = tf.data.Dataset.list_files(str(data_directory / '*.jpg'))
    ds = ds.cache()

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    ds_labeled = ds.map(read_images(str(data_directory), anchor_decode_func, other_decode_func),
                        num_parallel_calls=settings.AUTOTUNE)

    shuffle_buffer_size = file_count
    if repeat is not None:
        shuffle_buffer_size *= repeat
    ds_prepared = prepare_for_training(ds_labeled, shuffle=shuffle, batch_size=batch_size,
                                       shuffle_buffer_size=shuffle_buffer_size, repeat=repeat)

    return ds_prepared, step_per_epoch


def get_dataset_values(
        train_dir,
        test_dir,
        batch_size,
        repeat=None) -> (tf.data.Dataset, int, tf.data.Dataset, int):

    train_ds, train_steps_per_epoch = create_dataset(data_directory_name=train_dir,
                                                     batch_size=batch_size,
                                                     anchor_decode_func=simple_decode,
                                                     other_decode_func=simple_decode,
                                                     shuffle=True,
                                                     repeat=1)

    val_ds, _ = create_dataset(data_directory_name=test_dir,
                                                   batch_size=batch_size,
                                                   anchor_decode_func=simple_decode,
                                                   other_decode_func=simple_decode,
                                                   shuffle=False,
                                                   repeat=1)
    val_ds = val_ds.cache()

    test_ds = create_n_way_dataset(data_directory_name=test_dir,
                                                         batch_size=batch_size,
                                                         anchor_decode_func=simple_decode,
                                                         n_way_count=32)
    test_ds = test_ds.cache()

    return train_ds, train_steps_per_epoch, val_ds, test_ds


def base64_encode(image):
    return base64.b64encode(image).decode('utf-8')


def base64_decode(image):
    raw = np.frombuffer(base64.decodebytes(image))
    reshaped = raw.reshape(settings.IMG_HEIGHT, settings.IMG_WIDTH)
    return reshaped
