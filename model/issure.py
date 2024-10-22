import jax
from flax import nnx

class NNXDot(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
    self.w = nnx.Param(nnx.initializers.lecun_normal()(
      rngs.params(), (in_dim, out_dim)))
  def __call__(self, x: jax.Array):
    return x @ self.w

x = jax.random.normal(jax.random.key(42), (4, 32))
model = nnx.bridge.to_linen(NNXDot, 32, out_dim=64)
variables = model.init(jax.random.key(0), x)
y = model.apply(variables, x)

from flax import linen as nn
key = jax.random.PRNGKey(0)

tabulate_fn = nn.tabulate(
    model, key, compute_flops=True, compute_vjp_flops=True)
print(tabulate_fn(x))
