import os, sys, random, glob

import jax
import flax
import tensorflow as tf
import jax.numpy as jnp
from jax.experimental import jax2tf

sys.path.append("./model")
sys.path.append("./utils")
from utils import *
from model import TinyLPR


class RepresentativeDataset:
    def __init__(self, val_dir, input_shape=(1, 96, 192, 1), sample_size=200):
        self.input_shape = input_shape
        self.representative_list = random.sample(
            glob.glob(os.path.join(val_dir, '*.jpg')),
            sample_size,)

    def __call__(self):
        for image_path in self.representative_list:
            n, h, w, c = self.input_shape
            img = center_fit(
                cv2.cvtColor(cv2_imread(image_path), cv2.COLOR_BGR2GRAY),
                w, h, inter=cv2.INTER_AREA, top_left=True)
            img = np.reshape(img, self.input_shape).astype('float32') / 255.
            print(image_path)
            yield [img]


def jax2tflite(key, state, input_shape, dataset, save_path='model.tflite',
               inference_input_type=tf.uint8,
               inference_output_type=tf.uint8):

    def predict(input_img):
        return state.apply_fn({
            'params': state.params,
            'batch_stats': state.batch_stats
        }, input_img, train=False)

    tf_predict = tf.function(
        jax2tf.convert(predict, enable_xla=False),
        input_signature=[
            tf.TensorSpec(
                shape=list(input_shape),
                dtype=tf.float32,
                name='input_image')],
        autograph=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    if inference_input_type == tf.float32 and inference_output_type == tf.float32:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    else:
        converter.representative_dataset = dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    converter.inference_input_type = inference_input_type
    converter.inference_output_type = inference_output_type
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    with open('{}'.format(save_path), 'wb') as f:
        f.write(converter.convert())

    print('\033[92m[done]\033[00m Model converted to tflite.')


if __name__ == "__main__":
    IMG_SIZE = (1, 96, 192, 1)
    SAMPLE_SIZE = 1000
    VAL_DIR = "/Users/haoyu/Documents/datasets/lpr/val"
    val_ds = RepresentativeDataset(VAL_DIR, IMG_SIZE, SAMPLE_SIZE)

    import sys
    import yaml
    import optax
    sys.path.append("./")
    from fit import TrainState, load_ckpt

    cfg = yaml.safe_load(open("config.yaml"))

    key = jax.random.PRNGKey(0)
    x = jnp.zeros((1, *cfg["img_size"], 1), jnp.float32)

    model = TinyLPR(**cfg["model"])
    var = model.init(key, x, train=False)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(3e-4)
    )

    # state = load_ckpt(state, "weights/L16_9908")
    import orbax.checkpoint as ocp
    manager = ocp.PyTreeCheckpointer()
    state = manager.restore("weights/best", item=state)

    inout_type = tf.float32
    # inout_type = tf.uint8

    jax2tflite(key, state, IMG_SIZE, val_ds,
               save_path='model.tflite',
               inference_input_type=inout_type,
               inference_output_type=inout_type)
