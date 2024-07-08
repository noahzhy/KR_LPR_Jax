# Lightweight License Plate Recognition with JAX

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![JAX](https://img.shields.io/badge/JAX-0.4.25-blue)](https://github.com/google/jax) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/noahzhy/KR_LPR_TF)

This repository is a JAX implementations of lightweight license plate recognition (LPR) models.

## Data Preparation

The labeled data is required to train the model. The data should be organized as follows:

```dir
- data
  - labels.names
  - train
    - {license_plate_number}_{image_number}.jpg
    - {license_plate_number}_{image_number}.txt
    - ...
  - val
    - {license_plate_number}_{image_number}.jpg
    - {license_plate_number}_{image_number}.txt
    - ...
```

`license_plate_number` is the license plate number and make sure that the number is formatted like `12가1234`, `서울12가1234` and prepare a dict to parse the every character of the license plate number to the integer. The dict should be saved as [`labels.names`](data/labels.names) file. `image_number` is the number of the image and it is used to distinguish the same license plate number. The `.txt` file is the bounding boxes of each character in the license plate number. The format of the `.txt` file is as follows:

```txt
x1 y1 x2 y2
...
xn yn xn yn
```

The order of the bounding boxes should be the same as the order of the characters in the license plate number.

The dataloader will parse the data and convert the license plate characters to the integer using the `labels.names` file. The license plate images will be resized to `(64, 128)` or any other size you want. In addition, the mask of the license plate number will be created via the bounding boxes and the mask will be used to calculate the loss.

The losses of the model are as follows:
$$
L_{total} = \alpha * L_{FocalCTC} + \beta * L_{CenterCTC} + \gamma * L_{DiceBCE}, \quad \alpha = 1.5, \beta = 0.01, \gamma = 0.5
$$

$$
L_{CTC} = -\log \sum_{s \in S} P(s|X)
$$
$$
L_{FocalCTC} = \alpha (1 - P(s|X))^\gamma L_{CTC}, \quad \alpha = 0.8, \gamma = 3
$$

<!-- $$
L_{DiceBCE} = L_{Dice} + L_{BCE}
$$
$$
L_{Dice} = 1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}
$$
$$
L_{BCE} = -\frac{1}{N} \sum_{i=1}^N (Y_i \log(\hat{Y}_i) + (1 - Y_i) \log(1 - \hat{Y}_i))
$$

$$
L_{CenterCTC} = -\log \sum_{s \in S} P(s|X) \cdot \exp(-\frac{(x - \mu_x)^2}{2\sigma_x^2} - \frac{(y - \mu_y)^2}{2\sigma_y^2})
$$ -->

## Benchmark

|  Model    | Input Shape  |  Size  | Accuracy | Speed (ms) |
| --------- | ------------ | ------ | -------- | ----------:|
| tinyLPR-s | (64, 128, 1) | - KB   |  -       | - ms       |
| tinyLPR-l | (96, 192, 1) | 80 KB  |  0.9926  | 0.71 ms    |


## Ablation Study
