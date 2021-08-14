import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

def logslaterdet0(indices, x, L):
    """
        Compute the logarithm of the slater determinant of several plane-wave
    orbitals ln det(phi_j(r_i)), where phi_j(r_i) = 1/sqrt(L^dim) e^(i 2pi/L n_j r_i).

    INPUT SHAPES:
        indices: (n, dim) (array of integers)
        x: (n, dim)
    """
    #print("Call here!!!")
    k = 2*jnp.pi/L * indices
    k_dot_x = (k * x[:, None, :]).sum(axis=-1)
    _, dim = x.shape
    D = 1 / L**(dim/2) * jnp.exp(1j * k_dot_x)
    phase, logabsdet = jnp.linalg.slogdet(D)
    return logabsdet + jnp.log(phase)

#from functools import partial
#logslaterdet = partial(jax.custom_jvp, nondiff_argnums=(0, 2))(logslaterdet0)
logslaterdet = jax.custom_jvp(logslaterdet0)

@logslaterdet.defjvp
def logslaterdet_jvp(primals, tangents):
    """
        This implementation of the jvp rule makes use of specifics of slater
    determinants of plane-wave wavefunctions, thus is more efficient.
    """
    #print("Call here with jvp!!!")
    indices, x, L = primals
    _, dx, _ = tangents

    k = 2*jnp.pi/L * indices
    k_dot_x = (k * x[:, None, :]).sum(axis=-1)
    _, dim = x.shape
    D = 1 / L**(dim/2) * jnp.exp(1j * k_dot_x)

    phase, logabsdet = jnp.linalg.slogdet(D)
    primal_out = logabsdet + jnp.log(phase)

    k_dot_dx = (k * dx[:, None, :]).sum(axis=-1)
    tangent_out = (D * 1j * k_dot_dx * jnp.linalg.inv(D).T).sum()

    return primal_out, tangent_out
