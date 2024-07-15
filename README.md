# Lightweight License Plate Recognition with JAX

[![License](https://img.shields.io/badge/license-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0.html)
[![JAX](https://img.shields.io/badge/JAX-0.4.25-blue)](https://github.com/google/jax) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/noahzhy/KR_LPR)

This repository is a JAX implementations of lightweight license plate recognition (LPR) models.

## Data Preparation

The labeled data is required to train the model. The data should be organized as follows:

```dir
- data
  - labels.names
  - train
    - {license_plate_number}_{index}.jpg
    - {license_plate_number}_{index}.txt
    - ...
  - val
    - {license_plate_number}_{index}.jpg
    - {license_plate_number}_{index}.txt
    - ...
```

`license_plate_number` is the license plate number and make sure that the number is formatted like `12가1234`, `서울12가1234` and prepare a dict to parse the every character of the license plate number to the integer. The dict should be saved as [`labels.names`](data/labels.names) file. `image_number` is the number of the image and it is used to distinguish the same license plate number. The `.txt` file is the bounding boxes of each character in the license plate number. The format of the `.txt` file is as follows:

```txt
x1 y1 x2 y2
...
xn yn xn yn
```

The order of the bounding boxes should be the same as the order of the characters in the license plate number.

The dataloader will parse the data and convert the license plate characters to the integer using the `labels.names` file. The license plate images will be resized to `(96, 192)` or any other size you want. In addition, the mask of the license plate number will be created via the bounding boxes and the mask will be used to calculate the loss.

The losses of the model are as follows:

**For CTC**:

$$ L_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

$$ L_{center} = \sum_{i=1}^{n} \left(1 - \frac{c_i}{t}\right)^2 $$

$$ L_{CTC} = \alpha * L_{focal} + \beta * L_{center},L_{center} = \left\{
  \begin{aligned}
    0,            \quad\text{if } t \leq 20k    \\
    L_{center},   \quad\text{otherwise}         \\
  \end{aligned} 
\right. $$

**For Mask**:

$$ 
L_{Mask} = L_{Dice} + L_{BCE}
$$

## Benchmark

| Model   | Input Shape  | Size  | Accuracy | Speed (ms) |
| ------- | ------------ | ----- | -------- | ---------: |
| tinyLPR | (96, 192, 1) | 86 KB | 0.9908   |    0.44 ms |
