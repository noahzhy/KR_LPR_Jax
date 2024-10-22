import glob, random, time
from itertools import groupby

import jax
import yaml
import optax
import jax.numpy as jnp
import tensorflow as tf
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj, j2h
import tensorflow_datasets as tfds

from model.model import TinyLPR
from model.dataloader import get_data
from utils.utils import batch_ctc_greedy_decoder, batch_remove_blank
from fit import lr_schedule, fit, TrainState, load_ckpt


def load_dict(dict_path='data/labels.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = { i :h2j(_dict[i]) for i in range(len(_dict))}
    return _dict


label_dict = load_dict('data/labels.names')


def decode_label(pred, _dict=label_dict):
    # pred = np.argmax(pred, axis=-1)
    pred = [_dict[k] for k, g in groupby(pred) if k > 0]
    return "".join(pred)


@jax.jit
def predict(state: TrainState, batch):
    img, (_, label) = batch
    pred_ctc = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
        }, img, train=False)
    return pred_ctc, label


@jax.jit
def eval_step(state: TrainState, batch):
    pred_ctc, label = predict(state, batch)
    pred = batch_ctc_greedy_decoder(pred_ctc)
    # replace -1 with 0 in label and pred
    pred = jnp.where(pred == -1, 0, pred)
    label = jnp.where(label == -1, 0, label)
    ans = batch_array_comparison(pred, label, size=cfg["max_len"]+1)
    acc = jnp.mean(ans)
    return acc


def eval(key, model, input_shape, ckpt_dir, test_val):
    var = model.init(key, jnp.zeros(input_shape, jnp.float32), train=False)
    params = var["params"]
    batch_stats = var["batch_stats"]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.adam(1e-3),
    )

    state = load_ckpt(state, ckpt_dir)

    ds, _ = get_data(test_val, batch_size=32, data_aug=False)
    test_ds = tfds.as_numpy(ds)

    acc = []
    for batch in test_ds:
        a = eval_step(state, batch)
        acc.append(a)
    acc = jnp.stack(acc).mean()
    return acc


def single_test(key, model, input_shape, ckpt_dir, image_path):
    var = model.init(key, jnp.zeros(input_shape, jnp.float32), train=True)
    params = var["params"]
    batch_stats = var["batch_stats"]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(3e-4),
    )

    manager = ocp.PyTreeCheckpointer()
    state = manager.restore(ckpt_dir, item=state)

    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(
        img, (96, 192), method=tf.image.ResizeMethod.BILINEAR, antialias=True, preserve_aspect_ratio=True
    )
    img = tf.cast(img, tf.float32) / 255.
    img = tf.expand_dims(img, 0)

    (p_mask, _, pred_ctc), _ = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
        }, img, train=True, mutable=['batch_stats'])

    # pred_ctc = state.apply_fn({
    #     'params': state.params,
    #     'batch_stats': state.batch_stats
    #     }, img, train=False)

    # p_mask = jax.nn.sigmoid(p_mask)
    # p_mask = jnp.max(p_mask[:, :, :, :], axis=-1)
    # argmax
    p_mask = jnp.argmax(p_mask, axis=-1)
    p_mask = jnp.expand_dims(p_mask, axis=-1)

    plt.imshow(p_mask[0])
    plt.show()

    pred = batch_ctc_greedy_decoder(pred_ctc)
    return pred


if __name__ == "__main__":
    # cpu mode
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(0)
    cfg = yaml.safe_load(open("config.yaml"))
    model = TinyLPR(**cfg["model"])

    input_shape = (1, *cfg["img_size"], 1)
    ckpt_dir = "weights"

    test_val = "data/val.tfrecord"
    acc = eval(key, model, input_shape, ckpt_dir, test_val)
    print("\33[32mAvg acc: {:.4f}\33[00m".format(acc))

    # import glob, random

    # images = glob.glob("data/val/*.jpg")
    # random.shuffle(images)
    # image_path = images[0]
    # print(image_path)
    # pred = single_test(key, model, input_shape, ckpt_dir, image_path)
    # print(pred)
    # print(decode_label(pred[0]))
