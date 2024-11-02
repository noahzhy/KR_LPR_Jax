import os, sys, random, glob

import jax
import flax
from flax import nnx
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
        return state.apply_fn(params, other_variables)(input_img)[0]

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


def load_ckpt(model, ckpt_dir):
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        banner_message(["No checkpoint was loaded", "Training from scratch"])
        return model

    import orbax.checkpoint as ocp
    checkpointer = ocp.StandardCheckpointer()
    graphdef, abstract_state = nnx.split(model)

    ckpt_path = os.path.abspath(ckpt_dir)
    state_restored = checkpointer.restore(ckpt_path, abstract_state)
    model = nnx.merge(graphdef, state_restored)
    return model


if __name__ == "__main__":
    IMG_SIZE = (1, 96, 192, 1)
    SAMPLE_SIZE = 1000
    VAL_DIR = "/Users/haoyu/Documents/datasets/lpr/val"
    val_ds = RepresentativeDataset(VAL_DIR, IMG_SIZE, SAMPLE_SIZE)

    import sys
    import yaml
    import optax
    from flax.training import train_state
    sys.path.append("../")

    cfg = yaml.safe_load(open("config.yaml"))

    key = nnx.Rngs(0)
    x = jnp.zeros((1, *cfg["img_size"], 1), jnp.float32)
    model = TinyLPR(**cfg["model"], rngs=key)
    model = load_ckpt(model, "/Users/haoyu/Documents/Projects/LPR_Jax/weights/175")
    model.eval()

    graphdef, params, other_variables = nnx.split(model, nnx.Param, ...)

    class TrainState(train_state.TrainState):
        other_variables: nnx.State

    state = TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        other_variables=other_variables,
        tx=optax.adam(1e-3)
    )

    inout_type = tf.float32
    # inout_type = tf.uint8

    jax2tflite(key, state, IMG_SIZE, val_ds,
               save_path='model.tflite',
               inference_input_type=inout_type,
               inference_output_type=inout_type)
