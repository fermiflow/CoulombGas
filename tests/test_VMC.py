import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import haiku as hk
from VMC import make_logpsi_logp, logpsi_grad_laplacian

def test_kinetic_energy():
    """
        Test the present kinetic energy (i.e., laplacian) implementation, where
    the real and imaginary part are separated, yield correct result in the special
    case of identity flow.
    """
    n, dim = 7, 3
    L = 1.234

    identity = hk.transform(lambda x: x)
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    params = identity.init(key, x)

    from orbitals import sp_orbitals
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi, logp = make_logpsi_logp(identity, fake_manybody_indices, L)
    logpsix, grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, jnp.array([0]), logpsi)
    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

    print("kinetic energy:", kinetic)
    kinetic_analytic = (2*jnp.pi/L)**2 * (indices**2).sum()
    print("analytic result:", kinetic_analytic)
    assert jnp.allclose(kinetic, kinetic_analytic)
