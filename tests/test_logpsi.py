import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import haiku as hk
from orbitals import sp_orbitals
from logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logp

key = jax.random.PRNGKey(42)

def fermiflow(depth, spsize, tpsize, L, n, dim):
    from flow import FermiNet
    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, L)
        return model(x)
    flow = hk.transform(flow_fn)

    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = flow.init(key, x)
    return flow, x, params

"""
    Below two tests are meant to check the various transformation properties of 
the functions logpsi and logp.
"""
def test_logpsi():
    depth, spsize, tpsize, L = 3, 16, 16, 1.234
    n, dim = 7, 3
    flow, x, params = fermiflow(depth, spsize, tpsize, L, n, dim)

    sp_indices = jnp.array( sp_orbitals(dim)[0] )
    state_idx = jnp.array( np.random.choice(sp_indices.shape[0], size=n, replace=False))
    print("indices:\n", sp_indices[state_idx])

    logpsi = make_logpsi(flow, sp_indices, L)
    logpsix = logpsi(x, params, state_idx)

    print("---- Test ln Psi_n(x + R) = ln Psi_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsix_image = logpsi(x + image, params, state_idx)
    print("logpsix:", logpsix)
    print("logpsix_image:", logpsix_image)
    assert jnp.allclose(logpsix_image, logpsix)

    print("---- Test permutation invariance: Psi_n(Px) = +/- Psi_n(x) ----")
    P = np.random.permutation(n)
    logpsix_P = logpsi(x[P, :], params, state_idx)
    psix_P, psix = jnp.exp(logpsix_P), jnp.exp(logpsix)
    print("psix:", psix)
    print("psix_P:", psix_P)
    assert jnp.allclose(psix_P, psix) or jnp.allclose(psix_P, -psix)

def test_logp():
    depth, spsize, tpsize, L = 3, 16, 16, 1.234
    n, dim = 7, 3
    flow, x, params = fermiflow(depth, spsize, tpsize, L, n, dim)

    sp_indices = jnp.array( sp_orbitals(dim)[0] )
    state_idx = jnp.array( np.random.choice(sp_indices.shape[0], size=n, replace=False))
    print("indices:\n", sp_indices[state_idx])

    logpsi = make_logpsi(flow, sp_indices, L)
    logp = make_logp(logpsi)
    logpx = logp(x[None, ...], params, state_idx[None, ...])

    print("---- Test ln p_n(x + R) = ln p_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpx_image = logp(x[None, ...] + image, params, state_idx[None, ...])
    assert jnp.allclose(logpx_image, logpx)

    print("---- Test translation invariance: p_n(x + a) = p_n(x), where `a` is a common translation of all electrons ----")
    shift = jnp.array( np.random.randn(dim) )
    logpx_shift = logp(x[None, ...] + shift, params, state_idx[None, ...])
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
    params = identity.init(key, x)

    sp_indices = jnp.array( sp_orbitals(dim)[0] )
    state_idx = jnp.array( np.random.choice(sp_indices.shape[0], size=n, replace=False))
    print("indices:\n", sp_indices[state_idx])

    logpsi = make_logpsi(identity, sp_indices, L)
    _, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi)
    grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, state_idx[None, ...], key)
    assert grad.shape == (1, n, dim)
    assert laplacian.shape == (1,)

    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
    print("kinetic energy:", kinetic)
    kinetic_analytic = (2*jnp.pi/L)**2 * (sp_indices[state_idx]**2).sum()
    print("analytic result:", kinetic_analytic)
    assert jnp.allclose(kinetic, kinetic_analytic)

def test_laplacian():
    """ Check the two implementations of logpsi laplacian are equivalent. """
    depth, spsize, tpsize, L = 2, 4, 4, 1.234
    n, dim = 7, 3
    flow, x, params = fermiflow(depth, spsize, tpsize, L, n, dim)

    sp_indices = jnp.array( sp_orbitals(dim)[0] )
    state_idx = jnp.array( np.random.choice(sp_indices.shape[0], size=n, replace=False))
    print("indices:\n", sp_indices[state_idx])

    logpsi = make_logpsi(flow, sp_indices, L)
    _, logpsi_grad_laplacian1 = make_logpsi_grad_laplacian(logpsi)
    _, logpsi_grad_laplacian2 = make_logpsi_grad_laplacian(logpsi, forloop=False)
    grad1, laplacian1 = logpsi_grad_laplacian1(x[None, ...], params, state_idx[None, ...], key)
    grad2, laplacian2 = logpsi_grad_laplacian2(x[None, ...], params, state_idx[None, ...], key)
    assert jnp.allclose(grad1, grad2)
    assert jnp.allclose(laplacian1, laplacian2)

def test_laplacian_hutchinson():
    """
        Use a large batch sample to (qualitatively) check the Hutchinson estimator
    of the laplacian of logpsi.
    """
    depth, spsize, tpsize, L = 2, 4, 4, 1.234
    n, dim = 7, 3
    flow, x, params = fermiflow(depth, spsize, tpsize, L, n, dim)

    sp_indices = jnp.array( sp_orbitals(dim)[0] )
    state_idx = jnp.array( np.random.choice(sp_indices.shape[0], size=n, replace=False))
    print("indices:\n", sp_indices[state_idx])

    batch = 4000

    logpsi = make_logpsi(flow, sp_indices, L)
    logpsi_grad_laplacian = jax.jit(make_logpsi_grad_laplacian(logpsi)[1])
    grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, state_idx[None, ...], key)

    logpsi_grad_laplacian2 = jax.jit(make_logpsi_grad_laplacian(logpsi, hutchinson=True)[1])
    grad2, random_laplacian2 = logpsi_grad_laplacian2(
            jnp.tile(x, (batch, 1, 1)), params, jnp.stack([state_idx]*batch), key)
    laplacian2_mean = random_laplacian2.mean()
    laplacian2_std = random_laplacian2.std() / jnp.sqrt(batch)

    assert jnp.allclose(grad2, grad)
    print("batch:", batch)
    print("laplacian:", laplacian)
    print("laplacian hutchinson mean:", laplacian2_mean, "\tstd:", laplacian2_std)
