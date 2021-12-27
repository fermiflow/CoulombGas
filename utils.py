import jax
import jax.numpy as jnp

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    stacked_pytree = jax.tree_map(lambda x: jnp.stack([x] * num_devices), pytree)
    return shard(stacked_pytree)
