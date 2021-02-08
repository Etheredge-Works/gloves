from src import custom_model
from src import utils
import numpy as np
import tensorflow as tf
import pathlib

def test_create_model():
    pass


def test_run_model():
    assert False


def test_call():
    assert False


def test_partial_predict():
    #3data_dir = utils.get_dataset()
    data_dir = pathlib.Path(r"data\train")

    main_coons = list(data_dir.glob('*Maine_Coon*jpg'))
    #print('---------------------')
    all = list(data_dir.glob("*jpg"))

    #all_decoded = np.stack([utils.simple_decode(tf.io.read_file(str(file))) for file in all])

    not_main_coons = [item for item in all if item not in main_coons]
    #print(not_main_coons)
    not_main_coons_files = [str(file) for file in not_main_coons]
    not_main_coons_decoded = np.stack([utils.simple_decode(tf.io.read_file(file)) for file in not_main_coons_files])

    files = [str(file) for file in main_coons]
    #print(len(files))
    others = np.stack([utils.simple_decode(tf.io.read_file(file)) for file in files])

    net = custom_model.get_model()

    base_image = utils.simple_decode(tf.io.read_file("data\kitten_mittens.jpg"))  # TODO paramaterize
    base_image = utils.preprocess_input(base_image)
    bases = np.broadcast_to(base_image, others.shape)
    #bases = np.broadcast_to(custom_model.GlovesNet.base_image, others.shape)
    model_input = bases, others

    ms_predictions = net.predict(model_input, batch_size=32)
    pos_pred = np.round(ms_predictions)
    trues = sum(pos_pred)
    falses = len(pos_pred) - sum(pos_pred)
    mask = pos_pred == 1
    #assert np.all(mask)
    ms_average = np.average(pos_pred)

    bases = np.broadcast_to(base_image, not_main_coons_decoded.shape)
    predictions = net.predict([bases, not_main_coons_decoded], batch_size=32)
    neg_pred = np.round(predictions)
    mask = neg_pred == 0
    non_ms_average = np.average(neg_pred)
    assert ms_average > 0.9
    assert non_ms_average < 0.1
    #assert not np.all(mask)

