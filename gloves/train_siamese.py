import wandb

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
from pathlib import Path
from tensorflow.keras.layers import Dense
from dvclive.keras import DvcLiveCallback
import mlflow

from siamese.models import Encoder, SiameseModel, create_siamese_model
from siamese.data import create_dataset, get_labels_from_filenames, get_labels_from_files_path, create_n_way_dataset
from siamese.callbacks import NWayCallback
from utils import read_decode, random_read_decode
from models import build_custom_encoder, sigmoid_model
from utils.callbacks import MetricsCallback

from models.custom_model import L1DistanceLayer, L2DistanceLayer, CosineDistanceLayer

def log_summary(model, dir=None, name=None):
    name = name or model.name

    if dir:
        Path(dir).mkdir(parents=True, exist_ok=True)
        dir = dir + "/"
    filename = f"{dir}{name}.txt"

    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    mlflow.log_artifact(filename)



def train(
    train_dir: str, 
    train_extra_dir: str, 
    test_dir: str, 
    test_extra_dir: str, 
    out_model_path: str,
    out_encoder_path: str,
    out_metrics_path: str,
    out_summaries_path: str,
    *,  # only take kwargs for hypers
    #checkpoint_dir: str,
    height,
    width,
    depth,
    # hypers
    mutate_anchor,
    mutate_other,
    dense_reg_rate,
    conv_reg_rate,
    #activation,
    latent_nodes,
    final_activation,
    lr,
    optimizer,
    epochs,
    batch_size,
    verbose,
    eval_freq,
    reduce_lr_factor,
    reduce_lr_patience,
    early_stop_patience,
    mixed_precision,
    nway_freq,
    nways,
    use_batch_norm,
    loss,
    glob_pattern='*.jpg',
    nway_disabled=False,
    label_func='name',
    **_  # Other args in params file to ignore
):
    if label_func == 'name':
        label_func = get_labels_from_filenames
    elif label_func == 'path':
        label_func = get_labels_from_files_path
    else:
        raise ValueError


    if mixed_precision:
      policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
      tf.keras.mixed_precision.experimental.set_policy(policy)

    encoder = build_custom_encoder(
        input_shape=(height, width, depth),
        latent_nodes=latent_nodes,
        #activation=activation,
        final_activation=final_activation,
        dense_reg_rate=dense_reg_rate,
        conv_reg_rate=conv_reg_rate,
        use_batch_norm=use_batch_norm,
    )

    input1 = tf.keras.Input(encoder.output_shape[-1])
    input2 = tf.keras.Input(encoder.output_shape[-1])

    if loss == 'binary_crossentropy':
        outputs = Dense(1, activation='sigmoid', dtype='float32')(AbsDistanceLayer(dtype='float32')((input1, input2)))
        head = tf.keras.Model(inputs=(input1, input2), outputs=outputs, name='Distance')
        loss = 'binary_crossentropy'
        nway_comparator = 'max'
        metrics=['acc']
        monitor_metric = 'val_loss'
    else:
        if loss == 'l1':
            outputs = L1DistanceLayer(dtype='float32')((input1, input2))
        elif loss == 'l2':
            outputs = L2DistanceLayer(dtype='float32')((input1, input2))
        elif loss == 'cosine':
            #outputs = CosineDistanceLayer(dtype='float32')((input1, input2))
            outputs = CosineDistanceLayer(dtype='float32')((input1, input2))
        else:
            raise ValueError("Unknown loss: {loss}")

        head = tf.keras.Model(inputs=(input1, input2), outputs=outputs, name='NormDistance')
        loss = tfa.losses.ContrastiveLoss()
        nway_comparator = 'min'
        metrics=None
        monitor_metric = 'loss'

    model = create_siamese_model(encoder, head)
    log_summary(encoder, dir=out_summaries_path, name='encoder')
    log_summary(head, dir=out_summaries_path, name='head')
    log_summary(model, dir=out_summaries_path, name='model')
    
    from tensorflow.keras.optimizers import Adam
    optimizer_switch = {
        'adam': Adam
    }
    optimizer = optimizer_switch[optimizer]

    # TODO extract and pass in
    train_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(train_dir)/glob_pattern)))
    train_labels = tf.convert_to_tensor(label_func(train_files_tf))
    mlflow.log_param("train_dataset_size", len(train_labels))
    wandb.config.train_dataset_size = len(train_labels)
    assert len(train_files_tf) == len(train_labels)
    assert tf.size(train_files_tf) > 0, "no train files found"

    test_files_tf = tf.convert_to_tensor(tf.io.gfile.glob(str(Path(test_dir)/glob_pattern)))
    test_labels = tf.convert_to_tensor(label_func(test_files_tf))
    assert len(test_files_tf) == len(test_labels)
    assert tf.size(test_files_tf) > 0, "no test files found"
    mlflow.log_param("test_dataset_size", len(test_labels))
    wandb.config.test_dataset_size = len(test_labels)

    ds = create_dataset(
        anchor_items=train_files_tf,
        anchor_labels=train_labels,
        anchor_decode_func=random_read_decode if mutate_anchor else read_decode,
        other_decode_func=random_read_decode if mutate_other else read_decode,
        #other_items=extra_train_files,
        #other_labels=get_labels_from_files_path(extra_train_files),
        #repeat=1,
    ).batch(batch_size).prefetch(-1)

    # NOTE can just take from ds here to remove those items from the training set but leave the files available
    #ds = ds.take(1000)
    #ds = ds.map(lambda anchor, other, label: prepr)
    val_ds = create_dataset(
        anchor_items=test_files_tf,
        anchor_labels=test_labels,
        # anchor_items=train_files_tf,
        # anchor_labels=train_labels,
        anchor_decode_func=read_decode,
        # other_items=train_files_tf, # needed since test set won't have many items
        # other_labels=train_labels
    ).batch(batch_size).prefetch(-1) # TODO param cache
    # TODO should val_ds be cached? or should it change?

    if not nway_disabled:
        #assert False
        test_nway_ds = create_n_way_dataset(
            items=test_files_tf, 
            labels=test_labels,
            ratio=1.0, 
            anchor_decode_func=read_decode, 
            n_way_count=nways)

        nway_ds = create_n_way_dataset(
            items=train_files_tf, 
            labels=train_labels,
            ratio=0.1, 
            anchor_decode_func=read_decode, 
            n_way_count=nways)
    
    #mlflow.log_param("validation_dataset_size", len(list(val_ds))*batch_size)

    # TODO how can I use preprocessing layers? Dataset requires images to be the same size for batching...
    nway_callbacks = [] if nway_disabled else [
        NWayCallback(encoder=encoder, head=head, nway_ds=nway_ds, freq=nway_freq, comparator=nway_comparator, prefix_name="train_"),
        NWayCallback(encoder=encoder, head=head, nway_ds=test_nway_ds, freq=nway_freq, comparator=nway_comparator, prefix_name="test_")]
    callbacks=[
        ReduceLROnPlateau(monitor=monitor_metric, factor=reduce_lr_factor, patience=reduce_lr_patience),
        *nway_callbacks,
        EarlyStopping(monitor=monitor_metric, min_delta=0, patience=early_stop_patience, verbose=1, restore_best_weights=True),
        MetricsCallback(),
        wandb.keras.WandbCallback(),
    ]

    # TODO remove model from here and have it submit a post request to locally running rest api
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=metrics)
    print('Starting training')
    train_hist = model.fit(
        ds,
        epochs=epochs,
        #batch_size=batch_size,
        validation_data=val_ds,
        validation_freq=eval_freq,
        #steps_per_epoch=steps_per_epoch,
        #verbose=verbose,
        callbacks=callbacks
    )
    # TODO remove nway from sigmoid training

    history_dict = train_hist.history
    history_dict = {key: float(value[-1]) for key, value in history_dict.items()}
    #with open(metrics_file_name, 'w') as f:
        #yaml.dump(history_dict, f, default_flow_style=False)
    #print(history_dict)

    model.save(out_model_path, save_format='tf')
    mlflow.log_artifact(out_model_path)
    encoder.save(out_encoder_path, save_format='tf')
    mlflow.log_artifact(out_model_path)
    mlflow.log_artifact(out_metrics_path)
