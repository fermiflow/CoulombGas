import jax
import jax.numpy as jnp

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)
