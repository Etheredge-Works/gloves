#! python
from genericpath import exists
import os
os.environ['PYTHONHASHSEED']=str(4)
#import wandb
#wandb.init(project="gloves", config={"hyper":"parameter"})
import mlflow
mlflow.set_experiment("gloves")
from mlflow import pyfunc
import click
from pathlib import Path

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import utils
from settings import MIXED_PRECISION
import yaml
import tensorflow_addons as tfa
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

# TODO make sure test set is not mutated
# TODO make sure test set is not in trianing set

def log_metric(key, value, step=None):
    mlflow.log_metric(key=key, value=value, step=step)


@click.command()
# File stuff
@click.option('--train-dir', type=click.Path(exists=True), help='')
@click.option('--test-dir', type=click.Path(exists=True), help='')
@click.option('--all-dir', type=click.Path(exists=True), help='')
@click.option('--model-dir', type=click.Path(exists=False), help='')
@click.option('--model-filename', type=click.STRING, help='')
#@click.option('--encoder-dir', type=click.Path(exists=False), help='')
@click.option('--encoder-model', type=click.Path(exists=False), help='')
###############################################################################
# Image stuff
@click.option("--height", default=224, type=int)
@click.option("--width", default=224, type=int)
@click.option("--depth", default=3, type=int)
@click.option("--mutate-anchor", default=False, type=bool)
@click.option("--mutate-other", default=False, type=bool)
###############################################################################
# Model Stuff
@click.option('--dense-layers', default=2, type=click.INT, help='')
@click.option('--dense-nodes', default=512, type=click.INT, help='')
@click.option("--activation", default='relu', type=str)
@click.option("--latent-nodes", default=128, type=int)
@click.option("--dropout-rate", default=0.5, type=float)
@click.option("--final-activation", default='linear', type=str)
@click.option('--should-transfer-learn', type=click.BOOL, help='')
###############################################################################
# Training Stuff
@click.option('--loss_name', default='contrastive_loss', type=click.STRING, help='')
@click.option('--lr', type=click.FLOAT, help='')
@click.option('--optimizer', default='adam', type=click.STRING, help='')
@click.option('--epochs', type=click.INT, help='')
@click.option('--batch-size', type=click.INT, help='')
@click.option('--verbose', type=click.INT, help='')
#@click.option("--validation-ratio", default=0.3, type=float)
@click.option("--eval-freq", default=1, type=int)
@click.option("--reduce-lr-factor", default=0.8, type=float)
@click.option("--reduce-lr-patience", default=10, type=int)
@click.option("--early-stop-patience", default=10, type=int)
@click.option("--mixed-precision", default=False, type=bool)
@click.option("--unfreeze", default=0, type=int)
@click.option("--nway_freq", default=5, type=int)
@click.option("--nways", default=16, type=int)
def main(
        train_dir: str, 
        test_dir: str, 
        all_dir: str, 
        model_dir: str,
        model_filename: str,
        encoder_model: str,
        ########################
        height: int,
        width: int,
        depth: int,
        mutate_anchor: bool,
        mutate_other: bool,
        ########################
        dense_layers: int, 
        dense_nodes: int, 
        activation: str,
        latent_nodes: int,
        dropout_rate: float,
        final_activation: str,
        should_transfer_learn: bool,
        ########################
        loss_name: str,
        lr: float, 
        optimizer: str,
        epochs: int, 
        verbose: int,
        batch_size: int, 
        eval_freq: int,
        reduce_lr_factor: float,
        reduce_lr_patience: int,
        early_stop_patience: int,
        mixed_precision: bool,
        unfreeze: bool,
        nway_freq,
        nways,
        ):
    print(type(batch_size))
    assert(type(batch_size) == int)

    params = {
        'train_dir': train_dir, 
        'test_dir': test_dir, 
        'all_dir': all_dir, 
        'model_dir': model_dir,
        'dense_nodes': dense_nodes,
        'epochs': epochs, 
        'batch_size': batch_size, 
        'lr': lr, 
        'optmizer': optimizer, 
        'should_transfer_learn': should_transfer_learn, 
        'verbose': verbose, 
        'model_filename': model_filename,
    }
    mlflow.log_params(params)
    
    #custom_model.gridsearch()
    '''
    model, history = custom_model.create_model(
        #train_dir=pathlib.Path('data/images'),
        train_dir=train_dir,
        test_dir=test_dir,
        all_data_dir=all_dir,
        dense_nodes=dense_nodes, 
        epochs=epochs, 
        batch_size=batch_size, 
        lr=lr,
        optimizer=optimizer, 
        transfer_learning=transfer_learning,
        verbose=verbose,
        )
    '''
    #model = custom_model.glovesnet(dense_nodes, should_transfer_learn=transfer_learning)
    #print(model.summary())
    #model.compile(loss="binary_crossentropy",
                  #optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
    if mixed_precision != 0:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    trainable, encoder, _ = custom_model.build_model(
        dense_layers=dense_layers, 
        dense_nodes=dense_nodes, 
        latent_nodes=latent_nodes, 
        dropout_rate=dropout_rate, 
        input_shape=(height, width, depth), 
        should_transfer_learn=should_transfer_learn,
        activation=activation,
        final_activation=final_activation)
    encoder = siamese.models.``
    
    optimizer_switch = {
        'adam': tf.keras.optimizers.Adam
    }
    optimizer = optimizer_switch[optimizer]

    loss_switch = {
        'contrastive_loss': tfa.losses.ContrastiveLoss(),
        'mse': 'mse'
    }
    loss = loss_switch[loss_name]
    trainable.compile(loss=loss, optimizer=optimizer(lr=lr))

    # TODO extract and pass in
    train_ds, steps_per_epoch, val_ds, test_ds = utils.get_dataset_values(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    all_data_dir=all_dir,
                                                                    batch_size=batch_size)

    mlflow.log_param("dataset_size", len(list(train_ds)))
    mlflow.log_param("validation_dataset_size", len(list(val_ds)))

    #wandb.init(project="siamese")
    #model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=128, epochs=20, callbacks=[WandbCallback()])

    train_hist = trainable.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        validation_freq=eval_freq,
        #steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        shuffle=False,  # TODO dataset should handle shuffling
        callbacks=[
            ReduceLROnPlateau(monitor='loss', factor=reduce_lr_factor, patience=reduce_lr_patience),
            EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patience, verbose=1, restore_best_weights=True),
            #NWayCallback(),
            #WandbCallback(),
            #ModelCheckpoint(filepath=str(pathlib.Path('checkpoints/checkpoint')),save_weights_only=True, monitor='val_accuracy',
                                                #mode='max', save_best_only=True)
        ]
    )

    history_dict = train_hist.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    mlflow.log_metrics(history_dict)
    #with open(metrics_file_name, 'w') as f:
        #yaml.dump(history_dict, f, default_flow_style=False)
    #print(history_dict)

    #model.save("model", save_format='tf')
    Path(model_dir).mkdir(parents=True)
    trainable.save(str(Path(str(model_dir))/model_filename))
    #trainable.save(str(Path(str(model_dir))/encoder_model))
    encoder_path = Path(str(encoder_model))
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(str(encoder_path), save_format='h5')

    #mlflow.log_metrics(history_dict)

    #model, _ = create_model()
    #return model


if __name__ == "__main__":
    main()