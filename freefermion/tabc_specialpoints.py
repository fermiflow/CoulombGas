from mpmath import mpf, mp
from freefermion.analytic import Z_E

import numpy as np
import os

from specialpoints import Monkhorst_Pack

def _path(n, dim, Theta, Emax):
    return "/data/xiehao/CoulombGas/tabc/freefermion/analytic/" \
            + "n_%d_dim_%d_Theta_%f_Emax_%s/" % (n, dim, Theta, Emax)

def tabc_specialpoints(n, dim, Theta, Emax, Nk):

    path = _path(n, dim, Theta, Emax)
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Create directory: %s" % path)

    filename = os.path.join(path, "twists_Nk_%d.txt" % Nk)

    if not os.path.isfile(filename):
        print("Compute the analytic data from Monkhorst-Pack twist points.")
        f = open(filename, "w", buffering=1, newline="\n")

        twists, weights = Monkhorst_Pack(dim, Nk)

        for twist, weight in zip(twists, weights):
            twist = [mpf(twist_i) for twist_i in twist]
            F, E, S = Z_E(n, dim, mpf(str(Theta)), twist, Emax=Emax)
            print( ("weight: %f" + "\ttwist:" + "  %s"*dim + "\tF: %s \tE: %s \tS: %s") % (weight,
                            *[mp.nstr(twist_i) for twist_i in twist],
                            mp.nstr(F), mp.nstr(E), mp.nstr(S)) )
            f.write( ("%f" + "  %s"*dim + "  %s"*3 + "\n") % (weight,
                            *[mp.nstr(twist_i) for twist_i in twist],
                            mp.nstr(F), mp.nstr(E), mp.nstr(S)) )
        f.close()

    print("Read from the file %s" % filename)
    weight, *twist, F, E, S = np.loadtxt(filename, unpack=True)
    F_mean = (weight * F).sum()
    E_mean = (weight * E).sum()
    S_mean = (weight * S).sum()

    return F_mean, E_mean, S_mean

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analytic calculation of free fermions in the canonical ensemble using TABC. "
                                                 "The twist average is computed on Monkhorst-Pack grid points.")
    parser.add_argument("--n", type=int, default=37, help="total number of electrons")
    parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
    parser.add_argument("--Theta", type=float, default=0.15, help="dimensionless temperature T/Ef")
    parser.add_argument("--Emax", type=int, default=None, help="energy cutoff for the single-particle orbitals")
    parser.add_argument("--Nk", type=int, default=2, help="Monkhorst-Pack grid size")

    args = parser.parse_args()

    n, dim, Theta, Emax = args.n, args.dim, args.Theta, args.Emax
    Nk = args.Nk
    print("---- n = %d, dim = %d, Theta = %f, Emax = %s ----" % (n, dim, Theta, Emax))
    print("Monkhorst-Pack grid size Nk = %d" % Nk)

    analytics = tabc_specialpoints(n, dim, Theta, Emax, Nk)
    print("Analytic results for the thermodynamic quantities:\n"
            "F_mean: %f, E_mean: %f, S_mean: %f" % analytics)