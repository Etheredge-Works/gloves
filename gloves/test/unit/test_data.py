import os
# Don't want to use GPUs for unit tests as this will cause various machine issues
from .. import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src import custom_data
import pytest
from src import utils
from src import settings
import numpy as np

'''
def test_data():
    train_ds = custom_data.CatDogData.train_ds

    batches = train_ds.take(1)
    for (anchor_batch, other_batch), label_batch in batches:
        numpy_batch = anchor_batch.numpy()
        other = other_batch.numpy()
        label = label_batch.numpy()
        #3split_batch = tf.unstack(batch)
        for idx in range(numpy_batch.shape[0]):
            anchor = numpy_batch[idx]
            o = other[idx]
            assert not np.allclose(anchor, o)
'''
#@pytest.mark.parametrize("test_ratio", [0., 0.05, 0.1, 0.2, 0.5, 0.8, 0.9]) # 1.0 causes issues with shuffle
def test_data():
    train_ds, test_ds, steps_per_epoch, validation_steps = src.utils.get_dataset_values(
        settings.TRAIN_DIR,
        settings.TEST_DIR,
        batch_size=1,
        repeat=1)
    train_size = 0
    test_size = 0
    #count = train_ds.reduce(0, lambda x, _: x+1)
    for _ in train_ds:
        train_size += 1
    assert steps_per_epoch == train_size

    #test_size = 0
    for _ in test_ds:
        test_size += 1
    assert validation_steps == test_size

    total = test_size + train_size
    #assert round(total * test_ratio) == test_size


def test_shuffle():
    train_ds, test_ds, steps_per_epoch, validation_steps = src.utils.get_dataset_values(
        settings.TRAIN_DIR,
        settings.TEST_DIR,
        batch_size=1,
        repeat=2,
        mutate=False)

    items = train_ds.take(steps_per_epoch)
    #items_repeat = [train_ds.take(1) for step in range(steps_per_epoch)]
    items_repeat = train_ds.take(steps_per_epoch)
    #items_repeat = [next(train_ds) for step in range(steps_per_epoch)]
    #items_repeat = [next(train_ds) for step in range(steps_per_epoch)]
    count = 0
    for item_1, item_2 in zip(items, items_repeat):
        count += 1
        #for item_1, item_2 in zip(batch_1, batch_2):
        (anchor, other), label = item_1
        (anchor2, other2), label2 = item_2
        #anchor, other, label = batch_1.numpy()
        if np.array_equal(anchor, anchor2):
            assert not np.array_equal(other, other2)
        else:
            pass
                #assert np.array_equal(anchor, anchor2)



