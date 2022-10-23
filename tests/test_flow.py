import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk

import numpy as np

def generic_test_flow(model, L):
    """
        Generic test function for various flow models `model`.
    """
    n, dim = 7, 3
    key = jax.random.PRNGKey(42)
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, x)
    z = model.apply(params, None, x)

    # Test that flow results of two "equivalent" (under lattice translations of PBC)
    # particle configurations are equivalent.
    print("---- Test the flow is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    #print("image:", image / L)
    imagez = model.apply(params, None, x + image)
    assert jnp.allclose(imagez, z + image)

    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    #print("shift:", shift)
    shiftz = model.apply(params, None, x + shift)
    assert jnp.allclose(shiftz, z + shift)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    Pz = model.apply(params, None, x[P, :])
    assert jnp.allclose(Pz, z[P, :])

def test_FermiNet():
    from src.flow import FermiNet
    depth = 3
    spsize, tpsize = 16, 16
    L = 1.234
    
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)
    generic_test_flow(flow, L)
