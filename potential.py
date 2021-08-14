import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import erfc 

import numpy as np 

def kpoints(dim, Gmax):
    """
        Compute all the integer k-mesh indices (n_1, ..., n_dim) in spatial
    dimention `dim`, whose length do not exceed `Gmax`.
    """
    n = np.arange(-Gmax, Gmax+1)
    nis = np.meshgrid(*( [n]*dim ))
    G = np.array([ni.flatten() for ni in nis]).T
    G2 = (G**2).sum(axis=-1)
    G = G[(G2<=Gmax**2) * (G2>0)]
    return jnp.array(G)

def Madelung(dim, kappa, G):
    """
        The Madelung constant of a simple cubic lattice of lattice constant L=1
    in spatial dimension `dim`, namely the electrostatic potential experienced by
    the unit charge at a lattice site.
    """
    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    return g_k.sum() + g_0 - 2*kappa/jnp.sqrt(jnp.pi)

def Ewald(x, kappa, G, Vconst):
    """
        Total electrostatic energy (per cell) of a periodic system of lattice constant
    L=1, including both the coordinate-dependent and the constant Madelung part `Vconst`.
    """
    n, dim = x.shape

    i, j = jnp.triu_indices(n, k=1)
    rij = (x[:, None, :] - x)[i, j]
    rij -= jnp.rint(rij)
    # Only the nearest neighbor is taken into account in the present implementation of real-space summation.
    dij = jnp.linalg.norm(rij, axis=-1)
    V_shortrange = (erfc(kappa * dij) / dij).sum()

    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    V_longrange = ( g_k * jnp.cos(2*jnp.pi * G.dot(rij.T)).sum(axis=-1) ).sum() \
                    + g_0 * rij.shape[0]

    potential = V_shortrange + V_longrange + 0.5*n*Vconst
    return potential
