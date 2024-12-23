import os, sys, random, time

import jax
import jax.numpy as jnp
from flax import nnx


def _make_divisible(v, divisor, min_value=16):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class UpSample(nnx.Module):

    def __init__(self, in_feats, out_feats, rngs: nnx.Rngs):
        self.conv0 = nnx.Conv(
            in_features=in_feats, out_features=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn0 = nnx.BatchNorm(num_features=32, rngs=rngs)

        self.conv1 = nnx.Conv(
            in_features=32, out_features=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=64, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=64, out_features=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=128, rngs=rngs)

        self.last_conv = nnx.Conv(in_features=128, out_features=out_feats,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            kernel_init=nnx.initializers.kaiming_normal(),
            use_bias=True,
            rngs=rngs)

    def __call__(self, x):
        x = self.bn0(self.conv0(x))
        x = nnx.leaky_relu(x)
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="bilinear")
        x = self.bn1(self.conv1(x))
        x = nnx.leaky_relu(x)
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="bilinear")
        x = self.bn2(self.conv2(x))
        x = nnx.leaky_relu(x)
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="bilinear")
        return self.last_conv(x)


class BottleNeck(nnx.Module):

    def __init__(self, in_size, exp_size, out_size, s, k, activation, scale, rngs: nnx.Rngs):
        self.activation = activation

        _in = _make_divisible(in_size * scale, 8)
        exp = _make_divisible(exp_size * scale, 8)
        out = _make_divisible(out_size * scale, 8)
        self._in = _in
        self.out = out
        self.s = s

        self.conv0 = nnx.Conv(in_features=_in, out_features=exp,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn0 = nnx.BatchNorm(num_features=exp, rngs=rngs)

        self.conv1 = nnx.Conv(in_features=exp, out_features=exp,
            kernel_size=(k, k),
            strides=s,
            padding="same",
            feature_group_count=exp,
            use_bias=False,
            rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=exp, rngs=rngs)

        self.conv2 = nnx.Conv(in_features=exp, out_features=out,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=out, rngs=rngs)

    def __call__(self, inputs):
        x = self.activation(self.bn0(self.conv0(inputs)))
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.s == 1 and self._in == self.out:
            x = jnp.add(x, inputs)

        return x


class MobileNetV3Small(nnx.Module):

    def __init__(self, scale, out_channels, rngs: nnx.Rngs):
        # self.scale = scale

        self.conv0 = nnx.Conv(in_features=1, out_features=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn0 = nnx.BatchNorm(num_features=16, rngs=rngs)

        # # k  in    exp     out      NL          s
        # [3,  16,    16,     16,     nnx.relu,    2],
        # [3,  16,    72,     24,     nnx.relu,    1],
        # [3,  24,    88,     24,     nnx.relu,    1],
        # [5,  24,    96,     40,     nnx.relu,    2],
        # [5,  40,    240,    40,     nnx.relu,    1],
        # [5,  40,    120,    48,     nnx.relu,    1],
        # [5,  48,    144,    48,     nnx.relu,    1],
        self.bneck0 = BottleNeck(16, 16, 16, 2, 3, nnx.relu, scale, rngs)
        self.bneck1 = BottleNeck(16, 72, 24, 1, 3, nnx.relu, scale, rngs)
        self.bneck2 = BottleNeck(24, 88, 24, 1, 3, nnx.relu, scale, rngs)
        self.bneck3 = BottleNeck(24, 96, 40, 2, 5, nnx.relu, scale, rngs)
        self.bneck4 = BottleNeck(40, 240, 40, 1, 5, nnx.relu, scale, rngs)
        self.bneck5 = BottleNeck(40, 120, 48, 1, 5, nnx.relu, scale, rngs)
        self.bneck6 = BottleNeck(48, 144, 48, 1, 5, nnx.relu, scale, rngs)

        self.last = BottleNeck(16, 72, out_channels, 1, 5, nnx.relu, 1.0, rngs)

    def __call__(self, x):
        # 64, 128, 1
        x = nnx.relu(self.bn0(self.conv0(x)))
        x = self.bneck0(x)
        x = self.bneck1(x)
        x = self.bneck2(x)
        x = self.bneck3(x)
        x = self.bneck4(x)
        x = self.bneck5(x)
        x = self.bneck6(x)
        x = self.last(x)
        return x


class ChannelAttn(nnx.Module):

    def __init__(self, num_features, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=num_features,
            out_features=num_features,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_init=nnx.initializers.kaiming_normal(),
            rngs=rngs
        )

    def __call__(self, inputs):
        x = jnp.mean(inputs, axis=(1, 2), keepdims=True)
        x = self.conv(x)
        x = nnx.sigmoid(x)
        mul = jnp.multiply(x, inputs)
        return mul


class SpatialAttn(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=2,
            out_features=1,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_init=nnx.initializers.kaiming_normal(),
            rngs=rngs
        )

    def __call__(self, inputs):
        _min = jnp.min(inputs, axis=-1, keepdims=True)
        _max = jnp.max(inputs, axis=-1, keepdims=True)
        x = jnp.concatenate([_max, _min], axis=-1)
        x = self.conv(x)
        x = nnx.sigmoid(x)
        mul = jnp.multiply(x, inputs)
        return mul


class Attention(nnx.Module):

    def __init__(self, channels, temporal, rngs: nnx.Rngs):
        self.channels = channels
        self.temporal = temporal
        self.channel_attention = ChannelAttn(channels, rngs)
        self.spatial_attention = SpatialAttn(rngs)
        self.char_map = nnx.Conv(in_features=channels, out_features=temporal,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            kernel_init=nnx.initializers.kaiming_normal(),
            use_bias=True,
            rngs=rngs
        )

    def __call__(self, inputs):
        _channel = self.channel_attention(inputs)
        _spatial = self.spatial_attention(_channel)

        # 1x1 conv
        char_map = self.char_map(inputs)
        n, h, w, c = char_map.shape

        x = jnp.reshape(char_map, (-1, h * w, self.temporal))
        y = jnp.reshape(_spatial, (-1, h * w, self.channels))
        out = jnp.einsum("ijk,ijl->ikl", x, y)

        return out, char_map


class TinyLPR(nnx.Module):

    def __init__(self, time_steps, n_class, n_feat, rngs: nnx.Rngs):
        self.backbone = MobileNetV3Small(0.25, n_feat, rngs)
        self.attention = Attention(n_feat, time_steps, rngs)
        self.dense = nnx.Linear(in_features=n_feat, out_features=n_class,
            kernel_init=nnx.initializers.kaiming_normal(),
            use_bias=True,
            rngs=rngs)

        self.up = UpSample(time_steps, time_steps, rngs)

    def __call__(self, inputs, train=False):
        x = self.backbone(inputs)
        mat, attn = self.attention(x)
        out = self.dense(mat)

        if train:
            attn = nnx.softmax(attn)
            attn = self.up(attn)
            return attn, mat, out

        ## out = nnx.log_softmax(out) ## this make exported model strucutre more complex
        out = jnp.log(nnx.softmax(out)) ## same but simpler, only for inference and export
        return out


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")
    key = nnx.Rngs(0)
    model = TinyLPR(time_steps=16, n_class=68, n_feat=64, rngs=key)
    x = jnp.zeros((1, 96, 192, 1))
    y = model(x)
    for i in y:
        print(i.shape)
