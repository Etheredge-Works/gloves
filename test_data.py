import os
# Don't want to use GPUs for unit tests as this will cause various machine issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import data
import tensorflow as tf
import pytest
import numpy as np

def test_data():
    train_ds = data.CatDogData.train_ds

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


    #print(type(anchor))

