#! python
import os
os.environ['PYTHONHASHSEED']=str(4)
#import wandb
#wandb.init(project="gloves", config={"hyper":"parameter"})
import mlflow
mlflow.set_experiment("my-experiment")
from mlflow import pyfunc
import click
from pathlib import Path


import settings
from settings import MIXED_PRECISION
import yaml
#import mlflow


import numpy as np
np.random.seed(4)

import tensorflow as tf
tf.random.set_seed(4)
import mlflow.tensorflow
mlflow.tensorflow.autolog()
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)
import pathlib

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
import custom_model
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

# %%

'''
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
'''


# %%

# `tf.data`
# timeit(train_ds)

# %%

# %%

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

# %%

import argparse


@click.command()
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--all-dir', type=click.Path(exists=True), help='')
@click.option('--model-dir', type=click.Path(exists=False), help='')
@click.option('--dense-nodes', help='')
@click.option('--epochs', help='')
@click.option('--lr', help='')
@click.option('--optimizer', help='')
@click.option('--transfer-learning', help='')
@click.option('--verbose', help='')
@click.option('--model-filename', help='')
def main(
        train_dir: str, 
        test_dir: str, 
        all_dir: str, 
        model_dir: str,
        dense_nodes: int, 
        epochs: int, 
        batch_size: int, 
        lr: float, 
        optimizer: str,
        transfer_learning: bool,
        verbose: int,
        model_filename: str
        ):
    #custom_model.gridsearch()
    model, history = custom_model.create_model(
        #train_dir=pathlib.Path('data/images'),
        train_dir=train_dir,
        test_dir=test_dir,
        all_data_dir=all_dir,
        dense_nodes=dense_nodes, epochs=epochs, batch_size=batch_size, lr=lr,
        optimizer=optimizer, transfer_learning=transfer_learning,
        verbose=verbose,
        model_file_name=model_filename)

    history_dict = history.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    #with open(metrics_file_name, 'w') as f:
        #yaml.dump(history_dict, f, default_flow_style=False)
    #print(history_dict)

    #model.save("model", save_format='tf')
    model.save(str(Path(model_dir)/model_filename))

    #model, _ = create_model()
    #return model


if __name__ == "__main__":
    main()