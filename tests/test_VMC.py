import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import haiku as hk
from VMC import make_logpsi_logp

"""
    Below two tests are meant to check the various transformation properties of 
the functions logpsi and logp.
"""
def test_logpsi():
    from flow import FermiNet
    depth = 3
    spsize, tpsize = 16, 16
    L = 1.234
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)

    n, dim = 7, 3
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    params = flow.init(key, x)

    from orbitals import sp_orbitals
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi_grad_laplacian, _ = make_logpsi_logp(flow, fake_manybody_indices, L)
    logpsi, _, _ = logpsi_grad_laplacian(x[None, ...], params, jnp.array([0]))

    print("---- Test ln Psi_n(x + R) = ln Psi_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsi_image, _, _ = logpsi_grad_laplacian(x[None, ...] + image, params, jnp.array([0]))
    print("logpsi:", logpsi)
    print("logpsi_image:", logpsi_image)
    assert jnp.allclose(logpsi_image, logpsi)

    print("---- Test permutation invariance: Psi_n(Px) = +/- Psi_n(x) ----")
    P = np.random.permutation(n)
    logpsi_P, _, _ = logpsi_grad_laplacian(x[None, P, :], params, jnp.array([0]))
    psi_P, psi = jnp.exp(logpsi_P), jnp.exp(logpsi)
    print("psi:", psi)
    print("psi_P:", psi_P)
    assert jnp.allclose(psi_P, psi) or jnp.allclose(psi_P, -psi)

def test_logp():
    from flow import FermiNet
    depth = 3
    spsize, tpsize = 16, 16
    L = 1.234
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)

    n, dim = 7, 3
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    params = flow.init(key, x)

    from orbitals import sp_orbitals
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    _, logp = make_logpsi_logp(flow, fake_manybody_indices, L)
    logpx = logp(x[None, ...], params, jnp.array([0]))

    print("---- Test ln p_n(x + R) = ln p_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpx_image = logp(x[None, ...] + image, params, jnp.array([0]))
    assert jnp.allclose(logpx_image, logpx)

    print("---- Test translation invariance: p_n(x + a) = p_n(x), where `a` is a common translation of all electrons ----")
    shift = jnp.array( np.random.randn(dim) )
    logpx_shift = logp(x[None, ...] + shift, params, jnp.array([0]))
    assert jnp.allclose(logpx_shift, logpx)

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

    logpsi_grad_laplacian, _ = make_logpsi_logp(identity, fake_manybody_indices, L)
    logpsi, grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, jnp.array([0]))
    assert logpsi.shape == (1,)
    assert grad.shape == (1, n, dim)
    assert laplacian.shape == (1,)

    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
    print("kinetic energy:", kinetic)
    kinetic_analytic = (2*jnp.pi/L)**2 * (indices**2).sum()
    print("analytic result:", kinetic_analytic)
    assert jnp.allclose(kinetic, kinetic_analytic)
