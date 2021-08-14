import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from orbitals import sp_orbitals
from slater import logslaterdet0, logslaterdet

def test_slaterdet():
    """
        Basic check of the logslaterdet primitive, including antisymmetry property
    and invariance under translations.
    """
    n, dim = 7, 3
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    assert indices.shape == (n, dim)
    L = 1.234

    x = jnp.array( np.random.randn(n, dim) )
    det = jnp.exp(logslaterdet(indices, x, L))

    # Test the antisymmetry property.
    P = np.random.permutation(n)
    Pdet = jnp.exp(logslaterdet(indices, x[P, :], L))
    print("det:", det, "Pdet:", Pdet)
    assert jnp.allclose(Pdet, det) or jnp.allclose(Pdet, -det)

    # Test the translational invariance of the slater determinant.
    shift = jnp.array( np.random.randn(dim) )
    shifted_det = jnp.exp(logslaterdet(indices, x + shift, L))
    additional_phase = jnp.exp(1j * (2*jnp.pi/L*indices).dot(shift).sum())
    assert jnp.allclose(shifted_det, additional_phase * det)
    assert jnp.allclose(jnp.linalg.norm(shifted_det), jnp.linalg.norm(det))

def test_logslaterdet_AD():
    """
        Check AD behaviors of `logslaterdet` primitive against the "generic"
    implementation `logslaterdet0` which explicitly differentiating through
    `jnp.linalg.slotdet`.
    """
    n, dim = 7, 3
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    assert indices.shape == (n, dim)
    L = 1.234

    x = jnp.array( np.random.randn(n, dim) )
    dx = jnp.array( np.random.randn(n, dim) )

    # jvp test.
    logdet0, dlogdet0 = jax.jvp(lambda x: logslaterdet0(indices, x, L), (x,), (dx,))
    logdet, dlogdet = jax.jvp(lambda x: logslaterdet(indices, x, L), (x,), (dx,))
    assert jnp.allclose(logdet, logdet0)
    #print("dlogdet:", dlogdet)
    #print("dlogdet0:", dlogdet0)
    assert jnp.allclose(dlogdet, dlogdet0)

    # (1st order) gradient test.
    grad_logslaterdet0 = jax.grad(lambda x: logslaterdet0(indices, x, L), holomorphic=True)
    grad_logslaterdet = jax.grad(lambda x: logslaterdet(indices, x, L), holomorphic=True)
    grad_x0 = grad_logslaterdet0(x+0j)
    grad_x = grad_logslaterdet(x+0j)
    #print("grad_x:", grad_x)
    #print("grad_x0:", grad_x0)
    assert jnp.allclose(grad_x, grad_x0)

    # (2nd order) hessian test, via hessian-vector product with a random tangent dx.
    grad_x0_again, hvp0 = jax.jvp(grad_logslaterdet0, (x+0j,), (dx+0j,))
    grad_x_again, hvp = jax.jvp(grad_logslaterdet, (x+0j,), (dx+0j,))
    assert jnp.allclose(grad_x0_again, grad_x0) and jnp.allclose(grad_x_again, grad_x)
    #print("hvp:", hvp)
    #print("hvp0:", hvp0)
    assert jnp.allclose(hvp, hvp0)

def test_logslaterdet_eigenstate():
    """
        Check the plane-wave slater determinants are eigenstates of the laplacian
    (i.e., the kinetic) operator, by using AD to compute the local energy
    nabla^2 psi(x) / psi(x).
    """
    n, dim = 7, 3
    sp_indices, _ = sp_orbitals(dim)
    indices = sp_indices[ list(np.random.choice(sp_indices.shape[0],
                                                size=(n,), replace=False)) ]
    print("indices:\n", indices)
    assert indices.shape == (n, dim)
    L = 1.234

    x = jnp.array( np.random.randn(n*dim) )

    def div(f):
        def div_f(x):
            def body_fun(x, basevec):
                _, tangent = jax.jvp(f, (x,), (basevec,))
                return (tangent * basevec).sum()
            eye = jnp.eye(x.shape[0], dtype=x.dtype)
            return jax.vmap(body_fun, (None, 0), 0)(x, eye).sum()
        return div_f

    grad = jax.grad(lambda x: logslaterdet(indices, x.reshape(n, dim), L), holomorphic=True)
    laplacian = div(grad)
    E_analytic = (2*jnp.pi/L)**2 * (indices**2).sum()
    E = - laplacian(x+0j) - (grad(x+0j)**2).sum()
    print("E_analytic:", E_analytic)
    print("E:", E)
    assert jnp.allclose(E, E_analytic)
