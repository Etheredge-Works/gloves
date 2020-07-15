#! python
import os
os.environ['PYTHONHASHSEED']=str(4)
import wandb
wandb.init(project="gloves", config={"hyper":"parameter"})
import mlflow
mlflow.set_experiment("my-experiment")
from mlflow import pyfunc


import settings
from settings import MIXED_PRECISION
import yaml
#import mlflow


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import streamlit as st
#import random
#random.seed(4)
import numpy as np
np.random.seed(4)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#st.title("Mittens and Dave similarity analaysis")
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
    with open('params.yaml', 'r') as f:
        d = yaml.safe_load(f)


    parser = argparse.ArgumentParser(description="""
    This script will train a siamese network
    """)
    parser.add_argument("--train_dir", default='data/train', help="")
    parser.add_argument("--test_dir", default='data/train', help="")
    parser.add_argument("--all_dir", default='data/cleaned_images', help="")
    parser.add_argument("--dense_nodes", default=d['dense_nodes'], help="number of dense nodes for encoder")
    parser.add_argument("--epochs", default=d['epochs'], help="Number of epochs to run")
    parser.add_argument("--batch_size", default=d['batch_size'], help="None")
    parser.add_argument("--lr", default=d['lr'], help="None")
    parser.add_argument("--optimizer", default=d['optimizer'], help="None")
    parser.add_argument("--transfer_learning", default=d['transfer_learning'], help="None")
    parser.add_argument("--verbose", default='1', help="None")
    parser.add_argument("--metrics_file_name", default='metrics.yaml', help="None")
    parser.add_argument("--model_file_name", default='model.h5', help="None")

    args = parser.parse_args()

    nodes = int(args.dense_nodes)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)

    optimizer = args.optimizer
    transfer_learning = args.transfer_learning
    train_dir = pathlib.Path(args.train_dir)
    test_dir = pathlib.Path(args.test_dir)
    all_dir = pathlib.Path(args.all_dir)
    model_file_name = args.model_file_name
    metrics_file_name = args.metrics_file_name

    verbose = int(args.verbose)

    #custom_model.gridsearch()
    model, history = custom_model.create_model(
        #train_dir=pathlib.Path('data/images'),
        train_dir=train_dir,
        test_dir=test_dir,
        all_data_dir=all_dir,
        dense_nodes=nodes, epochs=epochs, batch_size=batch_size, lr=lr,
        optimizer=optimizer, transfer_learning=transfer_learning,
        verbose=verbose,
        model_file_name=model_file_name,
        metrics_file_name=metrics_file_name)

    history_dict = history.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    with open(metrics_file_name, 'w') as f:
        yaml.dump(history_dict, f, default_flow_style=False)
    print(history_dict)

    #model.save("model", save_format='tf')
    model.save(model_file_name)

    #model, _ = create_model()
    #return model

