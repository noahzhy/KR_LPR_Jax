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
    def __init__(self, in_feats, out_feats, rngs: nnx.Rngs, method="nearest"):
        self.method = method
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
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method=self.method)
        x = self.bn1(self.conv1(x))
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method=self.method)
        x = self.bn2(self.conv2(x))
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method=self.method)
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
        x = self.conv0(inputs)
        x = self.bn0(x)
        x = self.activation(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)        

        if self.s == 1 and self._in == self.out:
            x = jnp.add(x, inputs)

        return x


class MobileNetV3Small(nnx.Module):

    def __init__(self, width_multiplier, out_channels, rngs: nnx.Rngs):
        self.width_multiplier = width_multiplier
        self.rngs = rngs
        self.bnecks = [
            # k  in    exp     out      NL          s
            [3,  16,    16,     16,     nnx.relu,    2],
            [3,  16,    72,     24,     nnx.relu,    1],
            [3,  24,    88,     24,     nnx.relu,    1],
            [5,  24,    96,     40,     nnx.relu,    2],
            [5,  40,    240,    40,     nnx.relu,    1],
            [5,  40,    120,    48,     nnx.relu,    1],
            [5,  48,    144,    48,     nnx.relu,    1],
        ]
        self.conv0 = nnx.Conv(in_features=1, out_features=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_init=nnx.initializers.he_normal(),
            use_bias=False,
            rngs=rngs)
        self.bn0 = nnx.BatchNorm(num_features=16, rngs=rngs)
        self.last = BottleNeck(16, 72, out_channels, 1, 5, nnx.relu, 1.0, rngs)

    def __call__(self, x):
        # 64, 128, 1
        x = self.conv0(x)
        x = self.bn0(x)
        x = nnx.relu(x)

        for i, (k, _in, exp, out, NL, s) in enumerate(self.bnecks):
            x = BottleNeck(_in, exp, out, s, k, NL, self.width_multiplier, self.rngs)(x)

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
        self.time_steps = time_steps
        self.n_class = n_class
        self.n_feat = n_feat
        self.rngs = rngs

        self.backbone = MobileNetV3Small(0.25, n_feat, rngs)
        self.attention = Attention(n_feat, time_steps, rngs)
        self.dense = nnx.Linear(
            in_features=n_feat,
            out_features=n_class,
            kernel_init=nnx.initializers.kaiming_normal(),
            use_bias=True,
            rngs=rngs
        )

        self.up = UpSample(time_steps, time_steps, rngs)

    def __call__(self, inputs, train=True):
        x = self.backbone(inputs)
        mat, attn = self.attention(x)
        out = self.dense(mat)

        if train:
            attn = nnx.softmax(attn)
            attn = self.up(attn)
            return attn, mat, out

        # out = nnx.log_softmax(out)
        return out


if __name__ == '__main__':
    # jax cpu
    jax.config.update("jax_platform_name", "cpu")
    model = TinyLPR(time_steps=16, n_class=68, n_feat=64, rngs=nnx.Rngs(0))
    x = jnp.zeros((1, 96, 192, 1))
    # y = model(x)

    # model.train()
    # Pass in the arguments, not an actual module
    model = nnx.bridge.to_linen(TinyLPR, time_steps=16, n_class=68, n_feat=64, rngs=nnx.Rngs(0))
    variables = model.init(jax.random.key(0), x)
    y = model.apply(variables, x)
    for i in y:
        print(i.shape)

    from flax import linen as nn
    key = jax.random.PRNGKey(0)
    tabulate_fn = nn.tabulate(
        model, key, compute_flops=True, compute_vjp_flops=True)
    print(tabulate_fn(x))

    # check model output


    # for i in y:
    #     print(i.shape)
