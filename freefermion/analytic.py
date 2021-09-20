from mpmath import mpf, mp
mp.dps = 800

def z_e(dim, L, beta, Emax=None):
    """
        The partition function and expected energy of a single particle in a cubic 
    box of dimension `dim` and size `L`, at inverse temperature `beta`.

        The infinite sum is truncated according to the energy cutoff `Emax`. When
    `Emax` takes the special value `None`, the infinite sum is evaluated exactly
    (to a given precision).
    """
    if Emax:
        from orbitals import sp_orbitals
        _, Es = sp_orbitals(dim, Emax)
        Es = [(2*mp.pi/L)**2 * E for E in Es]
        z = mp.fsum(mp.exp(-beta*E) for E in Es)
        e = mp.fsum(E*mp.exp(-beta*E) for E in Es) / z
    else:
        z_single_dim = mp.jtheta(3, 0, mp.exp(-beta * (2*mp.pi/L)**2))
        e_single_dim = mp.jtheta(3, 0, mp.exp(-beta * (2*mp.pi/L)**2), derivative=2) \
                            / (-4) * (2*mp.pi/L)**2 / z_single_dim
        z = z_single_dim**dim
        e = dim * e_single_dim
    return z, e

def Z_E(n, dim, Theta, Emax=None):
    """
        The partition function and relevant thermodynamic quantities of `n` free
    (spinless) fermions in dimension `dim` and temperature `Theta`, computed using
    recursion relations.

        `Theta` is measured relative to the Fermi energy corresponding to certain
    dimensionless density parameter rs. The resulting physical quantities with
    energy dimension, such as the expected energy `E` and free energy `F`, have
    unit Ry/rs^2.

        The argument `Emax` determine the energy cutoff used in evaluating the
    single-particle partition function. See function "z_e" for details.
    """
    if dim == 3:
        L = (mpf("4/3") * mp.pi * n)**mpf("1/3")
        beta = 1 / ((mpf("4.5") * mp.pi)**mpf("2/3") * Theta)
    elif dim == 2:
        L = mp.sqrt(mp.pi*n)
        beta = 1 / (4 * Theta)

    #print("L:", L, "\nbeta:", beta)

    zs, es = tuple(zip( *[z_e(dim, L, k*beta, Emax) for k in range(1, n+1)] )) 
    #print("zs:", zs)
    #print("es:", es)

    Zs = [mpf(1)]
    Es = [mpf(0)]
    for N in range(1, n+1):
        Z = mp.fsum( (-1)**(k-1) * zs[k-1] * Zs[N-k]
                     for k in range(1, N+1)
                   ) / N
        E = mp.fsum( (-1)**(k-1) * zs[k-1] * Zs[N-k] * (k * es[k-1] + Es[N-k])
                     for k in range(1, N+1)
                   ) / N / Z
        Zs.append(Z)
        Es.append(E)
    #print("Zs:", Zs)

    F = -mp.log(Zs[-1])/beta
    E = Es[-1]
    S = beta*(E - F)
    return F, E, S

if __name__ == "__main__":
    import os

    Thetas = mp.linspace(mpf("0.02"), mpf("0.6"), 59)
    dim = 2
    ns = [13, 25, 37, 49, 61]
    path = "/data1/xieh/CoulombGas/master/freefermion/analytic"

    for n in ns:
        print("n = %d" % n)
        filename = os.path.join(path, "n_%d_dim_%d.txt" % (n, dim))
        if os.path.isfile(filename):
            print("The freefermion data file %s already exists. Skip..." % filename)
            continue
        fp = open(filename, "w", buffering=1, newline="\n")
        fp.write("#Theta\tf\te\ts\n")
        for i, Theta in enumerate(Thetas):
            F, E, S = Z_E(n, dim, Theta)
            f, e, s = F/n, E/n, S/n
            fp.write( ("%s" + "\t%s"*3 + "\n") %
                        (mp.nstr(Theta),
                         mp.nstr(f), mp.nstr(e), mp.nstr(s)) )
            print("Theta: %s\tf: %s\te: %s\ts: %s" %
                        (mp.nstr(Theta),
                         mp.nstr(f), mp.nstr(e), mp.nstr(s)))
        fp.close()
