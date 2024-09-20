import yaml
import numpy as np
import tensorflow as tf

n_map_threads = tf.data.experimental.AUTOTUNE
tf.config.experimental.set_visible_devices([], "GPU")

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
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask, label, size


def reshape_fn(image, mask, label, size, time_step=TIME_STEPS, target_size=TARGET_SIZE):
    image = tf.reshape(image, [size[0], size[1], 3])
    mask = tf.reshape(mask, [size[0], size[1], time_step])
    return image, mask, label


def resize_image(image, mask, label, target_size=TARGET_SIZE, keep_ratio=False):
    method = tf.image.ResizeMethod.BILINEAR
    if keep_ratio:
        image = tf.image.resize(image, target_size, method=method, antialias=True, preserve_aspect_ratio=True)
        mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True)
    else:
        image = tf.image.resize(image, target_size, method=method, antialias=True)
        mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask, label


def random_resize(image, mask, label, target_size=TARGET_SIZE):
    return resize_image(image, mask, label, target_size, keep_ratio=tf.random.uniform(()) > 0.5)


def align_label(label, time_step=TIME_STEPS):
    if BLANK_ID == -1:
        aligned_label = tf.zeros(len(label) * 2)
        for i in range(len(label)):
            aligned_label = tf.tensor_scatter_nd_update(aligned_label, [[i * 2 + 1]], [label[i]])
        if time_step != len(aligned_label):
            aligned_label = tf.pad(aligned_label, [[time_step - 1 - len(aligned_label), 1]], 'CONSTANT')
            aligned_label = tf.tensor_scatter_nd_update(aligned_label, [[time_step - 1]], [-1])
        return aligned_label
    return tf.pad(label, [[0, time_step - len(label)]], 'CONSTANT')


def pad_image_and_mask(image, mask, label, time_step=TIME_STEPS, target_size=TARGET_SIZE):
    image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    mask = tf.image.resize_with_pad(mask, target_size[0], target_size[1])
    image = tf.image.rgb_to_grayscale(image)
    label = align_label(label, time_step)
    return image, mask, label


def adjust_gamma(image):
    gamma = np.random.uniform(1.0, 2.0)
    gain = np.random.uniform(0.7, 1.5)
    return tf.image.adjust_gamma(image, gamma, gain)


def data_augment(image, mask, label):
    image = adjust_gamma(image)
    image = tf.image.random_contrast(image, 0.2, 1.5)
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_saturation(image, 0.1, 2.0)
    image = tf.image.random_brightness(image, 0.3)
    if tf.random.uniform(()) > 0.5:
        image = tf.math.abs(1 - image)  # 颜色反转
    return tf.clip_by_value(image, 0, 1), mask, label


def get_data(tfrecord, batch_size=32, data_aug=True, n_map_threads=n_map_threads):
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type='ZLIB', num_parallel_reads=n_map_threads)
    ds_len = sum(1 for _ in dataset) // batch_size

    ds = dataset.map(decode_data, num_parallel_calls=n_map_threads)
    ds = ds.map(reshape_fn, num_parallel_calls=n_map_threads)

    if data_aug:
        ds = ds.map(data_augment, num_parallel_calls=n_map_threads)
        ds = ds.map(random_resize, num_parallel_calls=n_map_threads)
    else:
        ds = ds.map(lambda img, msk, lbl: resize_image(img, msk, lbl, keep_ratio=True), num_parallel_calls=n_map_threads)

    ds = ds.map(pad_image_and_mask, num_parallel_calls=n_map_threads)
    ds = ds.shuffle(4096).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return ds, ds_len


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    batch_size = 8
    time_steps = 16
    aug = False
    BLANK_ID = -1

    label = tf.constant([1, 2, 3, 4, 5, 6, 7])
    res = align_label(label, time_steps)
    print(res)
    label = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
    res = align_label(label, time_steps)
    print(res)

    quit()
