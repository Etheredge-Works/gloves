import pytest
import utils
import os
import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
def test_get_dataset():
    data_dir = utils.get_dataset()
    assert os.path.isdir(data_dir), "Image directory was not created"
    assert isinstance(data_dir, pathlib.Path), "Returned directory is not pathlib.Path"
    files = os.listdir(data_dir)
    assert len(files) > 0, "No files in image directory."
    if len(files) > 0:
        for file in files:
            assert file.endswith("jpg") # Don't want any of those dumb mat files
            assert os.path.getsize(data_dir/file) > 0, f"{file} is empty."

@pytest.mark.parametrize("file", [utils.DATA_DIR/file for file in os.listdir(utils.DATA_DIR)])
def test_get_pairs(file):
    #dataset_dir = utils.get_dataset()
    #for file in os.listdir(dataset_dir):
    old_label = utils.get_label(file)
    anchor_file, positive_file, label = utils.get_pairs(file)
    anchor_label = utils.get_label(anchor_file)
    assert anchor_label == old_label
    other_label = utils.get_label(positive_file)
    assert (anchor_label == other_label) == bool(label)
    print(anchor_file)
    print(positive_file)
    print(label)
    #assert anchor_label != negative_label, "Anchor and negative match."

    # TODO do we want to test that the files are different?
    assert tf.convert_to_tensor(str(anchor_file)) != positive_file, "Anchor and positive are the same image."

"""
def test_get_label():
    paths = [
        "testing/this/path/thingy/asdfasdf_label_0.jpg",
        "testing\\this\\path\\thingy\\asdfasdf_label_1.jpg",
        "/testing/this/path/thingy/asdfasdf_label_2.jpg"
    ]
    for idx, file in enumerate(paths):
        predicted_label = utils.get_label(file)
        assert predicted_label == f"label", f"Incorrect label: Got {predicted_label} instead of label{idx}"
"""


def test_simple_decode():
    pass


def test_decode_img():
    pass


if __name__ == "__main__":
    print("no")