import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from potential import kpoints, Madelung, psi

def test_kpoints():
    Gmax = 4
    for dim in (2, 3):
        G = kpoints(dim, Gmax)
        print("dim:", dim, "Gmax:", Gmax)
        print(G)
        print("G.shape:", G.shape)
        assert G.shape[1] == dim

def generic_test_ewald(dim, x):
    for kappa in range(5, 11):
        print("---- dim = %d, kappa = %d ----" % (dim, kappa))
        for Gmax in range(5, 16):
            G = kpoints(dim, Gmax)
            print("Gmax:", Gmax, "\t\tG.shape:", G.shape, end="\t\t")
            Vconst = Madelung(dim, kappa, G)
            print("Vconst:", Vconst, end="\t\t")
            potential = psi(x, kappa, G)
            print("potential:", potential)

def test_ewald_3D():
    n, dim = 19, 3
    x = jnp.array( np.random.uniform(0., 1., (n, dim)) )
    generic_test_ewald(dim, x)

def test_ewald_2D():
    n, dim = 13, 2
    x = jnp.array( np.random.uniform(0., 1., (n, dim)) )
    generic_test_ewald(dim, x)
