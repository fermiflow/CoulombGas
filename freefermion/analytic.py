from mpmath import mpf, mp
mp.dps = 1200

def z_e(dim, L, beta, twist, Emax=None):
    """
        The partition function and expected energy of a single particle in a cubic 
    box of dimension `dim` and size `L`, at inverse temperature `beta`.
        `twist` is a (scaled) twist angle, in which each of the `dim` components
    is a number in the range (-1/2, 1/2]. The zero twist (i.e., Gamma point)
    corresponds to the periodic boundary condition.

        The infinite sum is truncated according to the energy cutoff `Emax`. When
    `Emax` takes the special value `None`, the infinite sum is evaluated exactly
    (to a given precision).
    """
    if Emax:
        from orbitals import sp_orbitals
        sp_indices, _ = sp_orbitals(dim, Emax)
        Es = [(2*mp.pi/L)**2 *
                mp.fsum((index_i+twist_i)**2 for index_i, twist_i in zip(index, twist))
                for index in sp_indices]
        z = mp.fsum(mp.exp(-beta*E) for E in Es)
        e = mp.fsum(E*mp.exp(-beta*E) for E in Es) / z
    else:
        q = mp.exp(-beta * (2*mp.pi/L)**2)

        z_single_dim = [ mp.jtheta(3, 1j*twist_i*beta*(2*mp.pi/L)**2, q) for twist_i in twist ]
        z = mp.fprod(z_single_dim) * q**mp.fdot(twist, twist)
        e_single_dim = [ (mp.jtheta(3, 1j*twist_i*beta*(2*mp.pi/L)**2, q, derivative=2) / (-4)
                         -mp.jtheta(3, 1j*twist_i*beta*(2*mp.pi/L)**2, q, derivative=1) * 1j*twist_i)
                            * (2*mp.pi/L)**2 / z_single_dim_i
                         for twist_i, z_single_dim_i in zip(twist, z_single_dim) ]
        e = mp.fsum(e_single_dim) + (2*mp.pi/L)**2 * mp.fdot(twist, twist)
        z, e = mp.re(z), mp.re(e)

    return z, e

def Z_E(n, dim, Theta, twist, Emax=None):
    """
        The partition function and relevant thermodynamic quantities of `n` free
    (spinless) fermions in dimension `dim` and temperature `Theta`, computed using
    recursion relations.

        `Theta` is measured relative to the Fermi energy corresponding to certain
    dimensionless density parameter rs. The resulting physical quantities with
    energy dimension, such as the expected energy `E` and free energy `F`, have
    unit Ry/rs^2.

        The arguments `Emax` and `twist` are relevant to the evaluation of
    single-particle partition function. See function "z_e" for details.
    """
    if dim == 3:
        L = (mpf("4/3") * mp.pi * n)**mpf("1/3")
        beta = 1 / ((mpf("4.5") * mp.pi)**mpf("2/3") * Theta)
    elif dim == 2:
        L = mp.sqrt(mp.pi*n)
        beta = 1 / (4 * Theta)

    #print("L:", L, "\nbeta:", beta)

    zs, es = tuple(zip( *[z_e(dim, L, k*beta, twist, Emax) for k in range(1, n+1)] )) 
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