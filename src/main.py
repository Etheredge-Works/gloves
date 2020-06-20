#! python
import os

import settings
from settings import MIXED_PRECISION
import yaml
#import mlflow



os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import streamlit as st
import numpy as np

np.random.seed(4)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#st.title("Mittens and Dave similarity analaysis")
import tensorflow as tf
import pathlib
tf.random.set_seed(4)

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
import custom_model
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

# %%

import time

default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            st.write('.', end='')
    st.write()
    end = time.time()

    duration = end - start
    st.write("{} batches: {} s".format(steps, duration))
    st.write("{:0.5f} Images/s".format(settings.BATCH_SIZE * steps / duration))


# %%

# `tf.data`
# timeit(train_ds)

# %%

# %%

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

# %%

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script will train a siamese network
    """)
    parser.add_argument("--dense_nodes", default=128, help="number of dense nodes for encoder")
    parser.add_argument("--epochs", default=1, help="Number of epochs to run")
    parser.add_argument("--batch_size", default=32, help="None")
    parser.add_argument("--lr", default=0.00003, help="None")

    args = parser.parse_args()

    nodes = int(args.dense_nodes)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)


    #custom_model.gridsearch()
    model, history = custom_model.create_model(
        #train_dir=pathlib.Path('data/images'),
        train_dir=pathlib.Path('data/train'),
        test_dir=pathlib.Path('data/test'),
        dense_nodes=nodes, epochs=epochs, batch_size=batch_size, lr=lr)

    history_dict = history.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    with open('metrics.yaml', 'w') as f:
        yaml.dump(history_dict, f, default_flow_style=False)
    print(history_dict)

    #model.save("model", save_format='tf')
    model.save("model.h5")

    #model, _ = create_model()
    #return model

