import os, re, sys, glob, time, random
from itertools import groupby
# cpu mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from jamo import h2j, j2hcj, j2h

sys.path.append('./')
from utils.utils import *


def load_dict(dict_path='data/labels.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = { i :h2j(_dict[i]) for i in range(len(_dict))}
    return _dict


label_dict = load_dict('data/labels.names')


def is_valid_label(label: str):
    label = ''.join(label)
    _city = [
        '서울', '부산', '대구', '인천', '광주',
        '대전', '울산', '세종', '경기', '강원',
        '충북', '충남', '전북', '전남', '경북',
        '경남', '제주',
    ]
    _pattern = r'^[가-힣]{2}[0-9]{2}[가-힣]{1}[0-9]{4}|^[0-9]{2,3}[가-힣]{1}[0-9]{4}$'
    # is valid
    if re.match(_pattern, label):
        return label[:2].isdigit() or label[:2] in _city
    else:
        return False


def decode_label(pred, _dict=label_dict):
    pred = np.argmax(pred, axis=-1)[0]
    pred = [_dict[k] for k, g in groupby(pred) if k != 0]
    return "".join(pred)


class TFliteDemo:
    def __init__(self, model_path, size=(96, 192), blank=0, conf_mode="min"):
        self.size = size
        self.blank = blank
        self.conf_mode = conf_mode
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.dtype = self.input_details[0]['dtype']

    def inference(self, x):
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def preprocess(self, img_path):
        image = cv2_imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = center_fit(image, self.size[1], self.size[0], top_left=True)
        image = np.reshape(image, (1, *image.shape, 1))
        if self.dtype == np.float32:
            image = image.astype(self.dtype) / 255.
        else:
            image = image.astype(self.dtype)
        return image

    def get_confidence(self, pred, mode="min"):
        conf = []
        idxs = np.argmax(pred, axis=-1)
        values = np.max(pred, axis=-1)

        for idx, c in zip(idxs, values):
            if idx == self.blank: continue
            # exp it if dtype is float
            if self.dtype == np.float32:
                c = np.exp(c)
            else:
                c = c / 255.

            conf.append(c)

        if mode == "min":
            return np.min(conf)

        return np.mean(conf)

    def postprocess(self, pred):
        label = decode_label(pred, load_dict())
        # is_valid = is_valid_label(label)
        conf = self.get_confidence(pred[0], mode=self.conf_mode)
        conf = float('{:.4f}'.format(conf))
        return {
            'label': label,
            'conf': conf,
            # 'valid': is_valid,
        }


if __name__ == '__main__':
    num_samples = 1000
    img_size = (96, 192)
    # init and load model
    demo = TFliteDemo('model.tflite', size=img_size)

    # get random image
    val_path = "/Users/haoyu/Documents/datasets/lpr/val"
    # random seed
    random.seed(1)
    img_list = random.sample(glob.glob(os.path.join(val_path, '*.jpg')), num_samples)

    res_confs = []

    # warm up for 50 times
    for i in range(50):
        image = demo.preprocess(random.choice(img_list))
        pred = demo.inference(image)

    avg_time = []
    for img_path in img_list:
        image = demo.preprocess(img_path)
        # inference
        start = time.process_time()
        pred = demo.inference(image)
        end = time.process_time()
        avg_time.append((end - start) * 1000)
        # post process
        result = demo.postprocess(pred)
        result['image_path'] = img_path
        res_confs.append(result)

    # sort by confidence
    res_confs.sort(key=lambda x: x['conf'], reverse=True)

    correct_count = 0
    for result in res_confs:
        img_path = result['image_path']
        gt = h2j(os.path.basename(img_path).split('_')[0])
        label = result['label']

        if gt == label:
            correct_count += 1
        else:
            print("\33[91m[ Error ]\33[00m", end=' ')
            print("path: {:20s} \tlabel: {:20s} \tconf: {:.4f}".format(result['image_path'], result['label'], result['conf']))

    print('\33[92m[done]\33[00m avg time: {:.4f} ms'.format(np.mean(avg_time)))
    print('\33[92m[done]\33[00m accuracy: {:.4f}'.format(correct_count / len(res_confs)))
