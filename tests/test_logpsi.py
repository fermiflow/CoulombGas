import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import haiku as hk
from logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logp

def make_indices(n, dim):
    from orbitals import sp_orbitals
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    return indices

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
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = flow.init(jax.random.PRNGKey(42), x)

    indices = make_indices(n, dim)
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi = make_logpsi(flow, fake_manybody_indices, L)
    logpsix = logpsi(x, params, 0)

    print("---- Test ln Psi_n(x + R) = ln Psi_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsix_image = logpsi(x + image, params, 0)
    print("logpsix:", logpsix)
    print("logpsix_image:", logpsix_image)
    assert jnp.allclose(logpsix_image, logpsix)

    print("---- Test permutation invariance: Psi_n(Px) = +/- Psi_n(x) ----")
    P = np.random.permutation(n)
    logpsix_P = logpsi(x[P, :], params, 0)
    psix_P, psix = jnp.exp(logpsix_P), jnp.exp(logpsix)
    print("psix:", psix)
    print("psix_P:", psix_P)
    assert jnp.allclose(psix_P, psix) or jnp.allclose(psix_P, -psix)

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
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = flow.init(jax.random.PRNGKey(42), x)

    indices = make_indices(n, dim)
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi = make_logpsi(flow, fake_manybody_indices, L)
    logp = make_logp(logpsi)
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
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = identity.init(jax.random.PRNGKey(42), x)

    indices = make_indices(n, dim)
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi = make_logpsi(identity, fake_manybody_indices, L)
    logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)
    logpsi, grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, jnp.array([0]))
    assert logpsi.shape == (1,)
    assert grad.shape == (1, n, dim)
    assert laplacian.shape == (1,)

    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
    print("kinetic energy:", kinetic)
    kinetic_analytic = (2*jnp.pi/L)**2 * (indices**2).sum()
    print("analytic result:", kinetic_analytic)
    assert jnp.allclose(kinetic, kinetic_analytic)

def test_laplacian_hutchinson():
    """
        Use a large batch sample to (qualitatively) check the Hutchinson estimator
    of the laplacian of logpsi.
    """
    from flow import FermiNet
    depth = 2
    spsize, tpsize = 4, 4
    L = 1.234
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)

    batch = 40000
    n, dim = 7, 3
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    key = jax.random.PRNGKey(42)
    params = flow.init(key, x)

    indices = make_indices(n, dim)
    print("indices:\n", indices)
    fake_manybody_indices = jnp.array(indices)[None, ...]

    logpsi = make_logpsi(flow, fake_manybody_indices, L)
    logpsi_grad_laplacian = jax.jit(make_logpsi_grad_laplacian(logpsi))
    logpsix, grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, jnp.array([0]))

    logpsi_grad_laplacian_hutchinson = jax.jit(make_logpsi_grad_laplacian(logpsi, key=key))
    logpsix2, grad2, random_laplacian2 = logpsi_grad_laplacian_hutchinson(
            jnp.tile(x, (batch, 1, 1)), params, jnp.zeros(batch, dtype=np.int32))
    laplacian2_mean = random_laplacian2.mean()
    laplacian2_std = random_laplacian2.std() / jnp.sqrt(batch)

    assert jnp.allclose(logpsix2[0], logpsix)
    assert jnp.allclose(grad2[0], grad)
    print("batch:", batch)
    print("laplacian:", laplacian)
    print("laplacian hutchinson mean:", laplacian2_mean, "\tstd:", laplacian2_std)
