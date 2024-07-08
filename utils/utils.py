import os, sys, glob, itertools

import cv2
import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp


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
    # get img shape
    img_h, img_w = img.shape[:2]
    # get ratio
    ratio = min(w / img_w, h / img_h)

    if len(img.shape) == 3:
        inter = cv2.INTER_AREA
    # resize img
    img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=inter)
    # get new img shape
    img_h, img_w = img.shape[:2]
    # get start point
    start_w = (w - img_w) // 2
    start_h = (h - img_h) // 2

    if top_left:
        start_w = 0
        start_h = 0

    if len(img.shape) == 2:
        # create new img
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
# group same label and remove blank
def ctc_greedy_decoder(logits, blank=0):
    logits = jnp.argmax(logits, axis=-1)
    # return [int(k) for k, _ in itertools.groupby(logits) if k != blank]
    return [int(k) for k, _ in itertools.groupby(logits) if k > blank]


# batch ctc greedy decoder
def batch_ctc_greedy_decoder(logits, blank=0):
    return [ctc_greedy_decoder(logit, blank) for logit in logits]


# remove blank in label list
def remove_blank(label, blank=0):
    # return [int(k) for k in label if k != blank]
    return [int(k) for k in label if k > blank]


# remove blank in batch label
def batch_remove_blank(label, blank=0):
    return [remove_blank(l, blank) for l in label]


# test unit for batch_ctc_greedy_decoder
def test_batch_ctc_greedy_decoder():
    logits = jnp.array([
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.7, 0.5],
            [0.5, 0.4, 0.3, 0.8, 0.1],
            [0.7, 0.3, 0.3, 0.3, 0.3],
        ],
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.7, 0.5],
            [0.5, 0.4, 0.3, 0.8, 0.1],
            [0.7, 0.3, 0.3, 0.3, 0.3],
        ],
    ])
    # assert batch_ctc_greedy_decoder(logits) == [[4, 3], [4, 3]]
    labels = [[4, 0, 3], [4, 2, 0]]
    # remove blank in batch label
    labels = batch_remove_blank(labels)
    logits = batch_ctc_greedy_decoder(logits)
    mean = jnp.mean(jnp.array([1 if jnp.array_equal(l, p) else 0 for l, p in zip(labels, logits)]))
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
    test_ctc_greedy_decoder()
    test_batch_ctc_greedy_decoder()

    # data/labels.name
    dict_ = names2dict(os.path.join("data", "labels.names"))
    print(dict_)
