import jax
import jax.numpy as jnp

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    stacked_pytree = jax.tree_map(lambda x: jnp.stack([x] * num_devices), pytree)
    return shard(stacked_pytree)

if __name__ == "__main__":
    from jax.config import config
    config.update("jax_enable_x64", True)
    import numpy as np

    num_devices = 2
    if jax.device_count() != num_devices:
        raise ValueError("%d GPU devices is needed." % num_devices)
    param1 = jnp.array( np.random.randn(10) )
    param2 = {"w": jnp.array( np.random.randn(3, 4) ),
              "b": jnp.array( np.random.randn(4) )}
    print("param1:", param1, "\nshape:", param1.shape)
    print("param2:", param2,
          "\nshape:", jax.tree_map(lambda x: x.shape, param2))
    replicated_param1 = replicate(param1, num_devices)
    replicated_param2 = replicate(param2, num_devices)
    print("replicated param1:", replicated_param1,
          "\nshape:", replicated_param1.shape)
    print("replicated param2:", replicated_param2,
          "\nshape:", jax.tree_map(lambda x: x.shape, replicated_param2))
