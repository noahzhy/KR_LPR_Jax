from flax import nnx
import tensorflow as tf
import jax.numpy as jnp
from jax.experimental import jax2tf
import optax
# import flax.linen as nn
from flax.training import train_state

key = nnx.Rngs(0)

class ConvModel(nnx.Module):
    def __init__(self, in_features, out_features, rngs):
        self.conv0 = nnx.Conv(in_features, 16, kernel_size=(3, 3), rngs=rngs)
        self.dwconv0 = nnx.Conv(16, 16, kernel_size=(3, 3), feature_group_count=16, rngs=rngs)
        self.bn0 = nnx.BatchNorm(num_features=16, rngs=rngs)

        self.conv1 = nnx.Conv(16, 16, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=16, rngs=rngs)

        self.linear1 = nnx.Linear(16, out_features, rngs=rngs)
        self.act = nnx.log_softmax

    def __call__(self, x):
        x = self.conv0(x)
        x = self.dwconv0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.act(x)
        return x


key = nnx.Rngs(0)
model = ConvModel(1, 10, key)
model.eval()
graphdef, params, other_variables = nnx.split(model, nnx.Param, ...)

input_shape = (1, 28, 28, 1)

class TrainState(train_state.TrainState):
    other_variables: nnx.State

state = TrainState.create(
    apply_fn=graphdef.apply,
    params=params,
    other_variables=other_variables,
    tx=optax.adam(1e-3),
)

def predict(input_img):
    return state.apply_fn(params, other_variables)(input_img)[0]


tf_predict = tf.function(
    jax2tf.convert(predict, enable_xla=False),
    input_signature=[
        tf.TensorSpec(
            shape=input_shape,
            dtype=tf.float32,
            name='input_image')],
    autograph=False)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_predict.get_concrete_function()],
    tf_predict
)

converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.optimizations = [tf.lite.Optimize.DEFAULT]

save_path = 'line.tflite'
with open('{}'.format(save_path), 'wb') as f:
    f.write(converter.convert())

print('\033[92m[done]\033[00m Model converted to tflite.')

import jax; jax.print_environment_info()