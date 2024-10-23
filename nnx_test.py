from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np


class MLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)


class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.mlp = MLP(dim, rngs=rngs)
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.mlp(x)
        x = self.linear1(x)
        return self.linear2(x)


# Instantiate the model and show we can run it.
model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
y = model(jnp.ones((4,)))
print(y)

_, state = nnx.split(model)
# nnx.display(state)

checkpointer = ocp.StandardCheckpointer()
checkpointer.save(
    "/Users/haoyu/Documents/Projects/LPR_Jax/checkpoints/", state)


# abstract_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(model)
print('The abstract NNX state (all leaves are abstract arrays):')
# nnx.display(abstract_state)

state_restored = checkpointer.restore(
    "/Users/haoyu/Documents/Projects/LPR_Jax/checkpoints", abstract_state)
jax.tree.map(np.testing.assert_array_equal, state, state_restored)
print('NNX State restored: ')
# nnx.display(state_restored)

# The model is now good to use!
model = nnx.merge(graphdef, state_restored)

y = model(jnp.ones((4,)))
print(y)
