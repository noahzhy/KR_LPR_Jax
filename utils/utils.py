import os, sys, glob, itertools

import cv2
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from functools import partial


def cv2_imwrite(file_path, img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def load_image(image_path):
    img = cv2.cvtColor(cv2_imread(image_path), cv2.COLOR_BGR2RGB)
    return jnp.array(img, dtype=jnp.float32) / 255.


# center fit and support rgb img
def center_fit(img, w, h, inter=cv2.INTER_NEAREST, top_left=True):
    img_h, img_w = img.shape[:2]
    ratio = min(w / img_w, h / img_h)
    img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=inter)
    # get new img shape
    img_h, img_w = img.shape[:2]
    start_w = 0 if top_left else (w - img_w) // 2
    start_h = 0 if top_left else (h - img_h) // 2

    if len(img.shape) == 2:
        new_img = np.zeros((h, w), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w] = img
    else:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w, :] = img

    return new_img


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ctc greedy decoder
def ctc_greedy_decoder(logits, blank=0):
    logits = jnp.argmax(logits, axis=-1)
    return [int(k) for k, _ in itertools.groupby(logits) if k > blank] # if k != blank]


@jax.jit
def ctc_greedy_decoder(logits, blank=0):
    """
    CTC greedy decoder
    Args:
        logits: (batch, time, num_classes)
        blank: blank label
    Returns:
        decoded: (batch, time), including blank(-1)
    """
    logits = jnp.argmax(logits, axis=-1)
    # groupby in JAX-friendly way
    changes = jnp.concatenate([jnp.array([True]), logits[1:] != logits[:-1]])
    decoded = jnp.where(changes & (logits != blank), logits, -1)
    return decoded


# batch ctc greedy decoder
@jax.jit
def batch_ctc_greedy_decoder(logits, blank=0):
    fn_map = jax.vmap(ctc_greedy_decoder, in_axes=0, out_axes=0)
    return fn_map(logits)


@partial(jax.jit, static_argnums=(2,))
def array_equality(a, b, size=10):
    # tile the a to the (size, ) shape and fill the blank with -1
    a = jnp.pad(a, (0, size - a.shape[0]), constant_values=0)
    b = jnp.pad(b, (0, size - b.shape[0]), constant_values=0)
    idx_a = jnp.nonzero(a, size=size, fill_value=-1)[0]
    idx_b = jnp.nonzero(b, size=size, fill_value=-1)[0]
    ans = jnp.all(jnp.take(a, idx_a) == jnp.take(b, idx_b))
    return ans


@partial(jax.jit, static_argnums=(2,))
def batch_array_comparison(logits, targets, size=10):
    """
    Compare two arrays
    Args:
        logits: (batch, time)
        targets: (batch, time)
        size: the max length of the array+1, incase of the last element is not blank
    Returns:
        ans: (batch, ) list of bool, True if equal, False otherwise
    """
    fn_map = jax.vmap(array_equality, in_axes=(0, 0, None), out_axes=0)
    return fn_map(logits, targets, size)


# test unit for batch_ctc_greedy_decoder
def test_batch_ctc_greedy_decoder():
    logits = jnp.array([
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 4
            [0.5, 0.4, 0.3, 0.2, 0.1],  # blank
            [0.1, 0.2, 0.3, 0.7, 0.5],  # 3
            [0.5, 0.4, 0.3, 0.8, 0.1],  # 3
            # [0.7, 0.3, 0.3, 0.3, 0.3],  # blank
        ],
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 4
            [0.5, 0.4, 0.3, 0.2, 0.1],  # blank
            [0.1, 0.2, 0.8, 0.7, 0.5],  # 2
            [0.5, 0.4, 0.3, 0.7, 0.1],  # 3
            # [0.7, 0.3, 0.3, 0.3, 0.3],  # blank
        ],
    ])
    # [[4, 3], [4, 3]]
    labels = [[4, 0, 3, 0], [4, 2, 3, 0]]
    labels = jnp.array(labels)
    logits = batch_ctc_greedy_decoder(logits)
    logits = jnp.where(logits == -1, 0, logits)
    print("logits\n", logits)

    ans = batch_array_comparison(logits, labels)
    mean = jnp.mean(ans)
    print(mean)
    print('\033[92m[pass]\033[00m batch_ctc_greedy_decoder() test passed.')


# test unit for ctc greedy decoder
def test_ctc_greedy_decoder():
    logits = jnp.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.7, 0.5],
        [0.5, 0.4, 0.3, 0.8, 0.1],
        [0.7, 0.3, 0.3, 0.3, 0.3],
    ])
    assert ctc_greedy_decoder(logits) == [4, 3]
    print('\033[92m[pass]\033[00m ctc_greedy_decoder() test passed.')


def names2dict(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return {i: c for i, c in enumerate([l.strip() for l in lines])}


if __name__ == "__main__":
    # test_ctc_greedy_decoder()
    test_batch_ctc_greedy_decoder()

    # # data/labels.name
    # dict_ = names2dict(os.path.join("data", "labels.names"))
    # print(dict_)
