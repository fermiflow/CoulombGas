import numpy as np 

def subsets(k, Pmax, Ps):
    """
        Given a set of several items with "prices" specified by the list Ps, find
    all subsets of length k whose total price do not exceed Pmax.
    """
    Nelements = len(Ps)
    result = ( ((), 0), )
    for i in range(1, k+1):
        result_new = []
        for subset, Ptotal in result:
            next_idx = subset[-1] + 1 if subset else 0
            while (next_idx + k - i < Nelements):
                if sum(Ps[next_idx:next_idx+k-i+1]) <= Pmax - Ptotal:
                    result_new.append( (subset + (next_idx,), Ptotal + Ps[next_idx]) )
                next_idx += 1
        result = tuple(result_new)
    indices, Ptotals = zip( *sorted(result, key=lambda index_P: index_P[1]) )
    return indices, Ptotals

def sp_orbitals(dim, Emax=60):
    """
        Compute index (n_1, ..., n_dim) and corresponding energy n_1^2 + ... + n_dim^2
    of all single-particle plane wave in spatial dimension `dim` whose energy
    does not exceed `Emax`.
        This function only serves to set up the momentum grid, hence zero twist angle
    (i.e., PBC) is used.

    OUTPUT SHAPES:
        indices: (n_orbitals, dim), Es: (n_orbitals,)
        (n_orbitals stands for total number of single-particle plane wave orbitals
    that fulfill the criteria.)
    """
    n_max = int(np.floor(np.sqrt(Emax)))
    n = np.arange(-n_max, n_max+1)
    nis = np.meshgrid(*( [n]*dim ))
    indices = np.array([ni.flatten() for ni in nis]).T
    Es = (indices**2).sum(axis=-1)
    indices, Es = indices[Es<=Emax], Es[Es<=Emax]

    sort_idx = Es.argsort()
    indices, Es = indices[sort_idx], Es[sort_idx]
    return indices, Es

def twist_sort(indices, twist):
    """
        Reorder the single-particle indices according to the energy under a given
    twist angle.
    """
    indices_twist = indices + twist
    Es_twist = (indices_twist**2).sum(axis=-1)
    sort = Es_twist.argsort()
    indices_twist, Es_twist = indices_twist[sort], Es_twist[sort]
    return indices_twist, Es_twist

def manybody_orbitals(n, dim, Ecut):
    """
        Compute the many-body plane-wave indices of `n` (spinless) fermions
    in spatial dimension `dim` whose total energy does not exceed E0 + `Ecut`,
    where E0 is the ground-state energy.

    OUTPUT SHAPES:
        manybody_indices: (n_manybody_states, n, dim)
        manybody_Es: (n_manybody_states,)
        (n_manybody_states stands for total number of many-body states of `n` fermions
    that fulfil the energy cutoff criteria.)
    """
    indices, Es = sp_orbitals(dim)
    manybody_E0 = Es[:n].sum()
    manybody_indices, manybody_Es = subsets(n, manybody_E0 + Ecut, list(Es))
    manybody_indices, manybody_Es = np.array(manybody_indices), np.array(manybody_Es)

    # indices.shape: (n_orbitals, dim);
    # manybody_indices.shape: (n_manybody_states, n)
    # --> indices[manybody_indices, :].shape: (n_manybody_states, n, dim)
    manybody_indices = indices[manybody_indices, :]

    return manybody_indices, manybody_Es

if __name__ == "__main__":
    for dim in (2, 3):
        indices, Es = sp_orbitals(dim)
        print("Es:", Es, Es.shape)
        print("indices:\n", indices, indices.shape)

        print("---- Closed-shell (spinless) electron numbers in dim = %d ----" % dim)
        Ef = Es[0]
        for i in range(Es.size):
            if (Es[i] != Ef):
                print("n = %d, Ef = %d" % (i, Ef))
                Ef = Es[i]
        print("n = %d, Ef = %d" % (Es.size, Es[-1]))

    n, dim = 7, 3
    print("---- %d (spinless) electrons in dim = %d ----" % (n, dim))
    for Ecut in range(4):
        _, manybody_Es = manybody_orbitals(n, dim, Ecut)
        print("Ecut = %d: number of many-body states = %6d" % (Ecut, manybody_Es.size))

    n, dim = 33, 3
    print("---- %d (spinless) electrons in dim = %d ----" % (n, dim))
    for Ecut in range(3):
        _, manybody_Es = manybody_orbitals(n, dim, Ecut)
        print("Ecut = %d: number of many-body states = %6d" % (Ecut, manybody_Es.size))

    n, dim = 13, 2
    print("---- %d (spinless) electrons in dim = %d ----" % (n, dim))
    for Ecut in range(9):
        _, manybody_Es = manybody_orbitals(n, dim, Ecut)
        print("Ecut = %d: number of many-body states = %6d" % (Ecut, manybody_Es.size))

    n, dim = 25, 2
    print("---- %d (spinless) electrons in dim = %d ----" % (n, dim))
    for Ecut in range(9):
        _, manybody_Es = manybody_orbitals(n, dim, Ecut)
        print("Ecut = %d: number of many-body states = %6d" % (Ecut, manybody_Es.size))