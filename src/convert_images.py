import io
import os
import sys

import tensorflow as tf
import PIL

def main(argv):
    path_images = './images'
    filenames_src = tf.gfile.ListDirectory(path_images)
    for filename_src in filenames_src:
        stem, extension = os.path.splitext(filename_src)
        if (extension.lower() != '.jpg'): continue

        pathname_jpg = '{}/{}'.format(path_images, filename_src)
        with tf.gfile.GFile(pathname_jpg, 'rb') as fid:
            encoded_jpg = fid.read(4)
        # png
        if(encoded_jpg[0] == 0x89 and encoded_jpg[1] == 0x50 and encoded_jpg[2] == 0x4e and encoded_jpg[3] == 0x47):
            # copy jpg->png then encode png->jpg
            print('png:{}'.format(filename_src))
            pathname_png = '{}/{}.png'.format(path_images, stem)
            tf.gfile.Copy(pathname_jpg, pathname_png, True)
            PIL.Image.open(pathname_png).convert('RGB').save(pathname_jpg, "jpeg")
        # gif
        elif(encoded_jpg[0] == 0x47 and encoded_jpg[1] == 0x49 and encoded_jpg[2] == 0x46):
            # copy jpg->gif then encode gif->jpg
            print('gif:{}'.format(filename_src))
            pathname_gif = '{}/{}.gif'.format(path_images, stem)
            tf.gfile.Copy(pathname_jpg, pathname_gif, True)
            PIL.Image.open(pathname_gif).convert('RGB').save(pathname_jpg, "jpeg")
        elif(filename_src == 'beagle_116.jpg' or filename_src == 'chihuahua_121.jpg'):
            # copy jpg->jpeg then encode jpeg->jpg
            print('jpeg:{}'.format(filename_src))
            pathname_jpeg = '{}/{}.jpeg'.format(path_images, stem)
            tf.gfile.Copy(pathname_jpg, pathname_jpeg, True)
            PIL.Image.open(pathname_jpeg).convert('RGB').save(pathname_jpg, "jpeg")
        elif(encoded_jpg[0] != 0xff or encoded_jpg[1] != 0xd8 or encoded_jpg[2] != 0xff):
            print('not jpg:{}'.format(filename_src))

if __name__ == "__main__":
    sys.exit(int(main(sys.argv) or 0))