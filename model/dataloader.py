import os, sys, random, time, glob, math
import yaml

import jax
import tqdm
import numpy as np
from PIL import Image
import jax.numpy as jnp
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
n_map_threads = tf.data.experimental.AUTOTUNE


cfg = yaml.safe_load(open("config.yaml"))
TARGET_SIZE = cfg["img_size"]
BLANK_ID = cfg["blank_id"]
TIME_STEPS = cfg["time_steps"]


def decode_data(example):
    ds_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'size': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, ds_desc)

    image = tf.io.decode_raw(example['image'], tf.uint8)
    label = tf.io.decode_raw(example['label'], tf.int64)
    mask = tf.io.decode_raw(example['mask'], tf.int64)
    size = tf.io.decode_raw(example['size'], tf.int64)
    # convert to float32 and normalize
    image = tf.cast(image, tf.float32) / 255.
    return image, mask, label, size


def reshape_fn(image, mask, label, size, time_step=TIME_STEPS, target_size=TARGET_SIZE):
    image = tf.reshape(image, [size[0], size[1], 3])
    mask = tf.reshape(mask, [size[0], size[1], time_step])
    return image, mask, label


def resize_image(image, mask, label, target_size=TARGET_SIZE):
    image = tf.image.resize(
        image, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    mask = tf.image.resize(
        mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False
    )
    return image, mask, label


def resize_image_keep_ratio(image, mask, label, target_size=TARGET_SIZE):
    image = tf.image.resize(
        image, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True, preserve_aspect_ratio=True
    )
    mask = tf.image.resize(
        mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False, preserve_aspect_ratio=True
    )
    return image, mask, label


def random_resize_keep_ratio_or_not(image, mask, label, target_size=TARGET_SIZE):
    if tf.random.uniform(()) > 0.5:
        return resize_image_keep_ratio(image, mask, label, target_size)
    else:
        return resize_image(image, mask, label, target_size)


def align_label(label, time_step=TIME_STEPS):
    # if len(label) < TIME_STEPS//2:
    #     label = tf.pad(label, [[TIME_STEPS//2 - len(label), 0]], 'CONSTANT', 0)

    # T = tf.repeat(label, 2)
    # for i in range(len(label)):
    #     T = tf.tensor_scatter_nd_update(T, [[i*2 + 1]], [0])

    # T = tf.tensor_scatter_nd_update(T, [[time_step-1]], [-1])
    # return T

    if BLANK_ID == -1:
        _label = tf.zeros(len(label) * 2)
        for i in range(len(label)):
            _label = tf.tensor_scatter_nd_update(_label, [[i * 2+1]], [label[i]])

        if time_step == len(_label):
            return _label

        # pad 0 to the left and pad -1 only last one
        # [1, 2, 3, 4, 5] -> [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, -1]
        T = tf.pad(_label, [[time_step-1 - len(_label), 1]], 'CONSTANT', constant_values=0)
        T = tf.tensor_scatter_nd_update(T, [[time_step-1]], [-1])
        return T
    else:
        return tf.pad(label, [[0, time_step - len(label)]], 'CONSTANT', constant_values=0)


    # if BLANK_ID == -1:
    #     _label = tf.zeros(len(label) * 2 - 1)
    #     for i in range(len(label)):
    #         _label = tf.tensor_scatter_nd_update(_label, [[i * 2]], [label[i]])

    #     if time_step == len(_label):
    #         return _label

    #     # pad 0 to the left and pad -1 only last one
    #     # [1, 2, 3, 4, 5] -> [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, -1]
    #     T = tf.pad(_label, [[time_step-1 - len(_label), 1]], 'CONSTANT', constant_values=0)
    #     T = tf.tensor_scatter_nd_update(T, [[time_step-1]], [-1])
    #     return T
    # else:
    #     return tf.pad(label, [[0, time_step - len(label)]], 'CONSTANT', constant_values=0)


def align_mask(mask, time_step):
    ''' given mask shape (H, W, T), pad to (H, W, time_step) '''
    return tf.pad(mask, [[0, 0], [0, 0], [time_step - mask.shape[-1], 0]], 'CONSTANT')


def pad_image_mask(image, mask, label, time_step=TIME_STEPS, target_size=TARGET_SIZE):
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = tf.image.pad_to_bounding_box(image, 0, 0, target_size[0], target_size[1])
        mask = tf.image.pad_to_bounding_box(mask, 0, 0, target_size[0], target_size[1])

    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(image)
    label = align_label(label, time_step)
    return image, mask, label


def random_pad(image, mask, label, target_size=TARGET_SIZE):
    H, W = target_size
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    pad_h = H - h
    pad_w = W - w

    if pad_h == 0 and pad_w == 0:
        return image, mask, label

    if h < H:
        pad_h = tf.random.uniform((), 0, H - h, dtype=tf.int32)

    if w < W:
        pad_w = tf.random.uniform((), 0, W - w, dtype=tf.int32)

    # Pad image and mask
    image = tf.image.pad_to_bounding_box(image, pad_h, pad_w, H, W)
    mask = tf.image.pad_to_bounding_box(mask, pad_h, pad_w, H, W)
    return image, mask, label


def data_augment(image, mask, label):
    gamma = np.random.uniform(low=1.0, high=2, size=[1,])
    gain = np.random.uniform(low=0.7, high=1.5, size=[1,])
    image = tf.image.adjust_gamma(image, gamma[0], gain[0])
    image = tf.image.random_contrast(image, 0.2, 1.5)
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_saturation(image, 0.1, 2.0)
    image = tf.image.random_brightness(image, 0.3)
    # color inversion
    if tf.random.uniform(()) > 0.5: image = tf.math.abs(1 - image)
    # clip to [0, 1]
    image = tf.clip_by_value(image, 0, 1)
    return image, mask, label


def get_data(tfrecord, batch_size=32, data_aug=True, n_map_threads=n_map_threads):
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type='ZLIB', num_parallel_reads=n_map_threads)
    ds_len = sum(1 for _ in dataset) // batch_size
    # parse and preprocess
    ds = dataset.map(decode_data, num_parallel_calls=n_map_threads)
    ds = ds.map(reshape_fn, num_parallel_calls=n_map_threads)

    if data_aug:
        ds = ds.map(data_augment, num_parallel_calls=n_map_threads)
        ds = ds.map(random_resize_keep_ratio_or_not, num_parallel_calls=n_map_threads)
        ds = ds.map(random_pad, num_parallel_calls=n_map_threads)
    else:
        ds = ds.map(resize_image_keep_ratio, num_parallel_calls=n_map_threads)

    ds = ds.map(pad_image_mask, num_parallel_calls=n_map_threads)
    ds = ds.shuffle(4096, reshuffle_each_iteration=data_aug).batch(
        batch_size, drop_remainder=True, num_parallel_calls=n_map_threads
    ).prefetch(tf.data.experimental.AUTOTUNE)
    return ds, ds_len


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    key = jax.random.PRNGKey(0)
    batch_size = 8
    # img_size = TARGET_SIZE
    time_steps = 16
    aug = False

    BLANK_ID = -1

    # load dict from names file to dict
    with open("data/labels.names", "r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    names = {i: name for i, name in enumerate(names)}
    print(names)

    label = tf.constant([1, 2, 3, 4, 5, 6, 7])
    res = align_label(label, time_steps)
    print(res)
    label = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
    res = align_label(label, time_steps)
    print(res)

    quit()

    tfrecord_path = "/home/noah/datasets/lpr/val.tfrecord"
    # tfrecord_path = "data/val.tfrecord"
    ds, ds_len = get_data(tfrecord_path, batch_size, aug)
    dl = tfds.as_numpy(ds)

    for data in tqdm.tqdm(dl, total=ds_len):
        img, mask, label = data
        print(img.shape, mask.shape, label.shape)

        # save one image as test.jpg
        img = img[0] * 255
        img = np.squeeze(img, -1)
        img = Image.fromarray(np.uint8(img))
        img.save('test.jpg')

        for i in range(16):
            # save as i.png
            mask_ = mask[0][:,:,i] * 255
            mask_ = Image.fromarray(np.uint8(mask_))
            mask_.save(f'tmp/{i}.png')

        # sum the mask to one channel
        mask = mask[0] * 255
        mask = np.sum(mask, axis=-1)
        # save the mask as test.png
        mask = Image.fromarray(np.uint8(mask))
        mask.save('test.png')
        break
