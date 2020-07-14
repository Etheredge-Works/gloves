import io
import os
import sys

import tensorflow as tf
import PIL
import pathlib

# https://github.com/tensorflow/models/issues/2194
def main(data_dir, cleaned_dir_name):
    path_images =pathlib.Path(data_dir)
    new_path = pathlib.Path(cleaned_dir_name)
    if not new_path.exists():
        new_path.mkdir()
    count = 0

    filenames_src = tf.io.gfile.listdir(str(path_images))
    for filename_src in filenames_src:
        count += 1 
        stem, extension = os.path.splitext(filename_src)
        if (extension.lower() != '.jpg'): continue

        pathname_jpg = str(path_images/filename_src)
        new_pathname_jpg = str(new_path/filename_src)
        with tf.io.gfile.GFile(pathname_jpg, 'rb') as fid:
            encoded_jpg = fid.read(4)
        # png
        if(encoded_jpg[0] == 0x89 and encoded_jpg[1] == 0x50 and encoded_jpg[2] == 0x4e and encoded_jpg[3] == 0x47):
            # copy jpg->png then encode png->jpg
            print('png:{}'.format(filename_src))
            pathname_png = f'{path_images}{os.path.sep}{stem}.png'
            tf.io.gfile.copy(pathname_jpg, pathname_png, True)
            PIL.Image.open(pathname_png).convert('RGB').save(new_pathname_jpg, "jpeg")
        # gif
        elif(encoded_jpg[0] == 0x47 and encoded_jpg[1] == 0x49 and encoded_jpg[2] == 0x46):
            # copy jpg->gif then encode gif->jpg
            print('gif:{}'.format(filename_src))
            pathname_gif = f'{path_images}{os.path.sep}{stem}.gif'
            tf.io.gfile.copy(pathname_jpg, pathname_gif, True)
            PIL.Image.open(pathname_gif).convert('RGB').save(new_pathname_jpg, "jpeg")
        elif(filename_src == 'beagle_116.jpg' or filename_src == 'chihuahua_121.jpg'):
            # copy jpg->jpeg then encode jpeg->jpg
            print('jpeg:{}'.format(filename_src))
            pathname_jpeg = f'{path_images}{os.path.sep}{stem}.jpeg'
            tf.io.gfile.copy(pathname_jpg, pathname_jpeg, True)
            PIL.Image.open(pathname_jpeg).convert('RGB').save(new_pathname_jpg, "jpeg")
        elif(encoded_jpg[0] != 0xff or encoded_jpg[1] != 0xd8 or encoded_jpg[2] != 0xff):
            print('not jpg:{}'.format(filename_src))
        else:
            tf.io.gfile.copy(pathname_jpg, new_pathname_jpg, True)
    assert count > 0

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script will clean corrupted images
    """)
    parser.add_argument("--data_dir", help="")
    parser.add_argument("--cleaned_dir_name", help="")

    args = parser.parse_args()

    data = pathlib.Path(args.data_dir)
    clean = pathlib.Path(args.cleaned_dir_name)
    sys.exit(int(main(data, clean) or 0))