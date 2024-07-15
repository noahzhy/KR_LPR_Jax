import os, sys, random, time, glob, math, timeit
from shutil import copyfile

import tqdm
import numpy as np
import PIL.Image as pil
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj, j2h


def load_dict(dict_path='data/labels.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = {h2j(_dict[i]): i for i in range(len(_dict))}
    return _dict


label_dict = load_dict('data/labels.names')

# split label
# e.g. '63루3348' -> ['6', '3', '루', '3', '3', '4', '8']
# e.g. '서울12가1234' -> ['서울', '1', '2', '가', '1', '2', '3', '4']
# e.g. 'A123B123' -> ['A', '1', '2', '3', 'B', '1', '2', '3'] 
def split_label(label):
    k_tmp = []
    split_label = []
    for i in label:
        if i.isdigit():
            if len(k_tmp) > 0:
                split_label.append(''.join(k_tmp))
                k_tmp = []
            split_label.append(i)
        else:
            k_tmp.append(i)
    return split_label


# unit test for split_label
def test_split_label():
    label = '63루3348'
    assert split_label(label) == ['6', '3', '루', '3', '3', '4', '8']
    label = '서울12가1234'
    assert split_label(label) == ['서울', '1', '2', '가', '1', '2', '3', '4']
    label = 'A123B123'
    assert split_label(label) == ['A', '1', '2', '3', 'B', '1', '2', '3']
    # color print
    print("\033[92m[pass]\033[00m split_label() test passed.")


# gen label
def gen_label(img_path, label_dict=label_dict):
    img_name = os.path.basename(img_path).replace(' ', '').split('_')[0]
    label = split_label(img_name)
    txt_label = ''.join(label)

    for i, char in enumerate(label):
        label[i] = label_dict[h2j(char)]

    return txt_label, label


def draw_heatmaps(shape, bboxes):

    def bbox_areas_log_np(bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        area = (y_max - y_min + 1) * (x_max - x_min + 1)
        return np.log(area)

    def radius_ttf(bbox, h, w):
        # alpha = 0.54
        alpha = 1.0
        h_radiuses_alpha = int(h / 2.0 * alpha)
        w_radiuses_alpha = int(w / 2.0 * alpha)
        return max(0, h_radiuses_alpha), max(0, w_radiuses_alpha)

    def gaussian2D(shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6.0
        sigma_y = h / 6.0
        gaussian = gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        y, x = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)
        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[h_radius - top : h_radius + bottom, w_radius - left : w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    heat_map = np.zeros(shape, dtype=np.float32)

    for b in range(shape[0]):
        for bbox in bboxes[b]:
            bbox = np.asarray(bbox)
            area = bbox_areas_log_np(bbox)
            fake_heatmap = np.zeros((shape[1], shape[2]))
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                # compute heat map
                h_radius, w_radius = radius_ttf(bbox, h, w)
                ct = np.array([
                    (bbox[1] + bbox[3]) / 2,
                    (bbox[0] + bbox[2]) / 2,
                ], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_truncate_gaussian(fake_heatmap, ct_int, h_radius, w_radius)
                heat_map[b, :, :, 0] = np.maximum(heat_map[b, :, :, 0], fake_heatmap)

    return heat_map


# gen mask
def gen_mask(im_path):
    w, h = pil.open(im_path).size
    _, label = gen_label(im_path)
    _mask = np.zeros((h, w, len(label)), dtype=np.int32)
    bbox = np.loadtxt(im_path.replace('jpg', 'txt'), dtype=int)

    for i, box in enumerate(bbox):
        b0 = max(0, box[0])
        b1 = max(0, box[1])
        b2 = min(w, box[2])
        b3 = min(h, box[3])
        _mask[b1:b3, b0:b2, i] = 1

    return _mask, label


# show sample of mask
def show_mask(im_path):
    mask, label = gen_mask(im_path)
    plt.figure(figsize=(8, 1))
    for i in range(mask.shape[2] + 1):
        if i == 0:
            plt.subplot(1, mask.shape[2] + 1, i + 1)
            plt.imshow(pil.open(im_path))
            plt.title('original')
            plt.axis('off')
        else:
            plt.subplot(1, mask.shape[2] + 1, i + 1)
            plt.imshow(mask[:, :, i - 1], cmap='gray')
            plt.title(label[i - 1])
            plt.axis('off')

    plt.savefig('mask_sample.png')
    # plt.show()


if __name__ == "__main__":
    # test split_label
    test_split_label()

    im_path = '/home/ubuntu/datasets/lpr/val/*.jpg'
    im_path = glob.glob(im_path)
    random.shuffle(im_path)

    # # gen label
    # t = timeit.timeit(lambda: gen_label(im_path[0]), number=1000) / 1000
    # print(f'time: {t / 1000:.8f} ms')

    t = timeit.timeit(lambda: gen_mask(im_path[0]), number=1000) / 1000
    print(f'time: {t / 1000:.10f} ms')

    # mask, label, txt_label = gen_mask(im_path[0])
    # im = pil.open(im_path[0]).convert('RGB')
    # print(np.array(im).shape)
    # print(mask.shape)

    show_mask(im_path[0])
