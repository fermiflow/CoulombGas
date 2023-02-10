from mpmath import mpf, mp
from .analytic import Z_E

import numpy as np
import os

def _path(n, dim, Theta, Emax):
    return "/data/xiehao/CoulombGas/tabc/freefermion/analytic/" \
            + "n_%d_dim_%d_Theta_%f_Emax_%s/" % (n, dim, Theta, Emax)

def tabc(n, dim, Theta, Emax, Ntwists, Ntwists_finished):

    path = _path(n, dim, Theta, Emax)
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Create directory: %s" % path)

    filename = os.path.join(path, "twists.txt")

    if Ntwists > Ntwists_finished:
        print("Compute the analytic data from new twist samples.")
        f = open(filename, "w" if Ntwists_finished == 0 else "a",
                 buffering=1, newline="\n")

        for i in range(0 if Ntwists_finished == 0 else Ntwists_finished + 1, Ntwists + 1):
            twist = ([mpf(0)] * dim) if i == 0 else [mp.rand() - mpf("0.5") for _ in range(dim)]
            F, E, S = Z_E(n, dim, mpf(str(Theta)), twist, Emax=Emax)
            print( ("%6d" + "\ttwist:" + "  %s"*dim + "\tF: %s \tE: %s \tS: %s") % (i,
                            *[mp.nstr(twist_i) for twist_i in twist],
                            mp.nstr(F), mp.nstr(E), mp.nstr(S)) )
            f.write( ("%6d" + "  %s"*dim + "  %s"*3 + "\n") % (i,
                            *[mp.nstr(twist_i) for twist_i in twist],
                            mp.nstr(F), mp.nstr(E), mp.nstr(S)) )
        f.close()

    print("Read from the file %s" % filename)
    _, *twist, F, E, S = np.loadtxt(filename, unpack=True)
    F_pbc, E_pbc, S_pbc = F[0], E[0], S[0]
    F, E, S = F[1:], E[1:], S[1:]

    print("Ntwists:", F.size)

    F_tabc, F_tabc_std = F.mean(), F.std()
    E_tabc, E_tabc_std = E.mean(), E.std()
    S_tabc, S_tabc_std = S.mean(), S.std()

    return F_pbc, E_pbc, S_pbc, \
           F_tabc, F_tabc_std, E_tabc, E_tabc_std, S_tabc, S_tabc_std

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analytic calculation of free fermions in the canonical ensemble using TABC. "
                                                 "The twist average is computed by random sampling.")
    parser.add_argument("--n", type=int, default=37, help="total number of electrons")
    parser.add_argument("--dim", type=int, default=2, help="spatial dimension")
    parser.add_argument("--Theta", type=float, default=0.15, help="dimensionless temperature T/Ef")
    parser.add_argument("--Emax", type=int, default=None, help="energy cutoff for the single-particle orbitals")

    parser.add_argument("--Ntwists", type=int, default=10000, help="the total desired number of twist angles")
    # For computation from scratch or more data, set `Ntwists_finished` to zero or appropriate positive number.
    parser.add_argument("--Ntwists_finished", type=int, default=0, help="already obtained number of twist angles")

    args = parser.parse_args()

    n, dim, Theta, Emax = args.n, args.dim, args.Theta, args.Emax
    print("---- n = %d, dim = %d, Theta = %f, Emax = %s ----" % (n, dim, Theta, Emax))

    analytics = tabc(n, dim, Theta, Emax, args.Ntwists, args.Ntwists_finished)
    print("Analytic results for the thermodynamic quantities:\n"
            "F_pbc: %f, E_pbc: %f, S_pbc: %f\n"
            "F_tabc: %f, F_tabc_std: %f\n"
            "E_tabc: %f, E_tabc_std: %f\n"
            "S_tabc: %f, S_tabc_std: %f"
            % analytics)