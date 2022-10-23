from mpmath import mpf, mp

def thermo_quantities(dim, Theta):
    """
        Thermodynamic quantities of free (spinless) fermions in dimension `dim` and
    temperature `Theta`, computed in thermodynamic limit.

    OUTPUT:
        z: fugacity
        f, e, s: free energy, energy and entropy density (per particle), respectively.
            f and e are in unit Ry/rs^2.
    """
    d = mpf(dim)
    z = mp.findroot(lambda z: mp.gamma(d/2+1) * mp.polylog(d/2, -z) * Theta**(d/2) + 1, 10)
    epsilon_F = 4 * mp.gamma(d/2+1)**(4/d)
    e = d/2 * mp.polylog(d/2+1, -z) / mp.polylog(d/2, -z) * Theta * epsilon_F
    s = (d/2+1) * mp.polylog(d/2+1, -z) / mp.polylog(d/2, -z) - mp.log(z)
    f = e - Theta * epsilon_F * s
    return z, f, e, s

if __name__ == "__main__":
    import os
    dim = 2
    Thetas = mp.linspace(mpf("0.02"), mpf("0.60"), 59)

    path = "/data1/xieh/CoulombGas/master/freefermion/analytic"
    filename = os.path.join(path, "n_inf_dim_%d.txt" % dim)
    if os.path.isfile(filename):
        print("The freefermion data file %s already exists. Skip..." % filename)
        exit(0)

    fp = open(filename, "w", buffering=1, newline="\n")
    fp.write("#Theta\tf\te\ts\n")

    for Theta in Thetas:
        _, f, e, s = thermo_quantities(dim, Theta)
        f, e, s = f.real, e.real, s.real
        fp.write( ("%s" + "\t%s"*3 + "\n") %
                    (mp.nstr(Theta),
                     mp.nstr(f), mp.nstr(e), mp.nstr(s)) )
        print("Theta: %s\tf: %s\te: %s\ts: %s" %
                    (mp.nstr(Theta),
                     mp.nstr(f), mp.nstr(e), mp.nstr(s)))

    fp.close()
