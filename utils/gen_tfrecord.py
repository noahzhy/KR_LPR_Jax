import os, sys, glob

import yaml
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.append("./utils")
from gen_label import *
from utils import load_image

cfg = yaml.safe_load(open("config.yaml"))
TIME_STEPS = cfg["time_steps"]


def gen_mask(bbox, size, len_label, time_step=TIME_STEPS):
    mask = np.zeros((size[0], size[1], time_step), dtype=np.int32)
    h, w = size

    for i, box in enumerate(bbox):
        b0 = max(0, box[0])
        b1 = max(0, box[1])
        b2 = min(w, box[2])
        b3 = min(h, box[3])
        # mask[b1:b3, b0:b2, i] = 1
        mask[b1:b3, b0:b2, time_step-(2*(len_label-i)-1)] = 1

        # hmap = draw_heatmaps((1, h, w, 1), [[box]])
        # mask[:, :, time_step-(2*(len_label-i)-1)] = hmap[:, :, :, 0]

    return mask


# resize image to width=128 and keep aspect ratio, also resize the bboxes(4 points)
# bbox: [x1, y1, x2, y2] and int64
def resize_image_keep_aspect_ratio(image, bbox, width=192):
    h, w, _ = image.shape
    ratio = width / w
    new_h = int(h * ratio)
    image = tf.image.resize(image, (new_h, width), antialias=True)
    bbox = tf.cast(bbox, tf.float32)
    bbox = tf.cast(tf.round(bbox * ratio), tf.int32)
    return image, bbox


def resize_image_and_bbox(image, bbox, size=(96, 192)):
    h, w, _ = image.shape
    r_h = size[0] / h
    r_w = size[1] / w
    image = tf.image.resize(image, size, antialias=True)
    bbox = tf.cast(bbox, tf.float32)
    bbox = tf.cast(tf.round(bbox * [r_w, r_h, r_w, r_h]), tf.int32)
    return image, bbox


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_tfrecord(dir_path, file_name):
    writer = tf.io.TFRecordWriter(
        'data/{}.tfrecord'.format(file_name),
        options=tf.io.TFRecordOptions(compression_type='ZLIB'))

    img_ds = glob.glob(dir_path + '/*.jpg')

    for img_path in tqdm.tqdm(img_ds):
        txt_path = img_path.replace('.jpg', '.txt')
        bbox = np.loadtxt(txt_path, dtype=np.int32)
        _, label = gen_label(img_path)

        image = Image.open(img_path).convert('RGB')
        image, bbox = resize_image_keep_aspect_ratio(np.array(image, dtype=np.float32), bbox)
        height, width, _ = image.shape
        mask = gen_mask(bbox, (height, width), len(label))

        # # sum the mask to one channel
        # print('shape:', image.shape, mask.shape, len(label))
        # mask = np.sum(mask, axis=-1)
        # mask = Image.fromarray(np.array(mask * 255, dtype=np.uint8))
        # mask.save('test.png')

        # # draw box on the image
        # image = Image.fromarray(np.array(image, dtype=np.uint8))
        # draw = ImageDraw.Draw(image)

        # for box in bbox:
        #     draw.rectangle([box[0], box[1], box[2], box[3]], outline='red')

        # # img.show()
        # # save the image
        # image.save('test.jpg')
        # quit()

        image = np.array(image, dtype=np.uint8).tobytes()
        mask = np.array(mask, dtype=np.int32).tobytes()
        label = np.array(label, dtype=np.int32).tobytes()
        size = np.array([height, width], dtype=np.int32).tobytes()

        feature = {
            'image': _bytes_feature(image),
            'mask': _bytes_feature(mask),
            'label': _bytes_feature(label),
            'size': _bytes_feature(size),
        }

        writer.write(tf.train.Example(features=tf.train.Features(
            feature=feature)).SerializeToString())

    writer.close()
    print("\033[1;32m{} tfrecord done\033[0m".format(file_name))


if __name__ == '__main__':
    # tmp_test = '/Users/haoyu/Documents/datasets/lpr/tmp_test'
    val_path = '/Users/haoyu/Documents/datasets/lpr/val'
    test_path = '/Users/haoyu/Documents/datasets/lpr/test'
    train_path = '/Users/haoyu/Documents/datasets/lpr/train'

    # gen_tfrecord(tmp_test, 'tmp_test')
    gen_tfrecord(val_path, 'val')
    gen_tfrecord(test_path, 'test')
    gen_tfrecord(train_path, 'train')
    print('done')
