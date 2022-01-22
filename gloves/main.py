#! python
from genericpath import exists
import os
import dvclive
import psutil
import mlflow
import wandb
import tensorflow as tf
from tensorflow.python.framework.tensor_conversion_registry import get
print(tf.version.GIT_VERSION, tf.version.VERSION)
from pathlib import Path
from train_siamese import train

os.environ['PYTHONHASHSEED']=str(4)
from mlflow import pyfunc
import click

# TODO test tensorboard for profiling
import yaml
import numpy as np
np.random.seed(4)
tf.random.set_seed(4)
import mlflow.tensorflow
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)
import pathlib

#from siamese.layers import NormDistanceLayer

import tensorflow as tf
import numpy as np



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


# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set
def log_metric(key, value, step=None):
    wandb.log({key: value}, step=step)
    mlflow.log_metric(key=key, value=value, step=step)



from copy import deepcopy
def mlflow_log_wrapper(func):
    def inner(*args, **kwargs):
        params = deepcopy(kwargs)
        #for arg in args:
            #arg_name = f'{arg=}'.split('=')[0]
            #params[arg_name] = arg

        if params['loss'] == 'binary_crossentropy':
            name = 'gloves-sigmoid' 
            sub_name = 'sigmoid'
        else: 
            sub_name = params['loss'].split('_')[0]
            name = f'gloves-{sub_name}-distance'
        #wandb.init(project=name, config=params)
        wandb.init(project="gloves", config=dict(
            type=sub_name,
            **params)
            )
        mlflow.set_experiment("gloves")
        mlflow.log_params(dict(type=sub_name, **params))

        return func(*args, **kwargs)
    return inner





@click.command()
# File stuff
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--train-extra-dir', default=None, type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--test-extra-dir',  default=None, type=click.Path(exists=True), help='')
@click.option('--param-path',  default=None, type=click.Path(exists=True), help='')
@click.option('--param-parent-key',  default='train', type=click.STRING, help='')
@click.option('--out-model-path', type=click.Path(exists=None), help='')
@click.option('--out-encoder-path', type=click.Path(exists=None), help='')
@click.option('--out-metrics-path', type=click.Path(exists=None), help='')
@click.option('--out-summaries-path', type=click.Path(exists=None), help='')
@click.option("--loss", default='cosine', type=str)
@click.option("--glob-pattern", default="*.jpg", type=str)
@click.option("--nway-disabled", default=False, type=bool)
@click.option("--label-func", default='name', type=str)
@mlflow_log_wrapper
def main(
        train_dir: str, 
        train_extra_dir: str, 
        test_dir: str, 
        test_extra_dir: str, 
        param_path: str, 
        param_parent_key: str,
        out_model_path: str,
        out_encoder_path: str,
        out_metrics_path: str,
        out_summaries_path: str,
        loss: str,
        glob_pattern: str,
        nway_disabled: bool,
        label_func: str,
):

    mlflow.tensorflow.autolog(every_n_iter=1)

    dvclive.init(out_metrics_path, summary=True, html=True)
    with open(param_path, "r") as f:
        train_kwargs = yaml.safe_load(f)[param_parent_key]

    train(
        train_dir=train_dir,
        train_extra_dir=train_extra_dir,
        test_dir=test_dir,
        test_extra_dir=test_extra_dir,
        loss=loss,
        out_model_path=out_model_path,
        out_encoder_path=out_encoder_path,
        out_metrics_path=out_metrics_path,
        out_summaries_path=out_summaries_path,
        glob_pattern=glob_pattern,
        nway_disabled=nway_disabled,
        label_func=label_func,
        **train_kwargs)


if __name__ == "__main__":
    limit_gpu_memory_use()
    main()
