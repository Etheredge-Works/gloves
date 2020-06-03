import pathlib

import numpy as np
import tensorflow as tf

import utils
from utils import read_images, prepare_for_training


class CatDogData:
    raw_data_dir = utils.get_dataset()
    data_dir = pathlib.Path(raw_data_dir)
    all_ds = tf.data.Dataset.list_files(str(data_dir / '*.jpg'))
    image_count = len(list(data_dir.glob('*.jpg')))
    test_size = int(image_count * utils.TEST_RATIO)
    real_count = image_count - int(image_count * utils.TEST_RATIO)
    # test_size = np.ceil(image_count * TEST_RATIO)
    test_ds = all_ds.take(test_size)
    list_ds = all_ds.skip(test_size)

    test_set_size = np.floor(image_count * utils.TEST_RATIO)
    STEPS_PER_EPOCH = np.ceil((image_count - test_set_size) / utils.BATCH_SIZE)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    # labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    labeled_ds = list_ds.map(read_images(utils.decode_img, utils.simple_decode), num_parallel_calls=utils.AUTOTUNE)
    test_labeled_ds = test_ds.map(read_images(utils.simple_decode, utils.simple_decode),
                                  num_parallel_calls=utils.AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds, shuffle=True, shuffle_buffer_size=real_count)
    test_ds = prepare_for_training(test_labeled_ds, shuffle=False)

    @staticmethod
    def data():
        return CatDogData.train_ds, CatDogData.test_ds

    # %%