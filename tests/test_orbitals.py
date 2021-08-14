from orbitals import sp_orbitals, manybody_orbitals

def test_sp_orbitals():
    for dim in (2, 3):
        indices, Es = sp_orbitals(dim)
        print("single-particle plane-wave orbitals in dim = %d:" % dim)
        print("indices:\n", indices, indices.shape)
        print("Es:\n", Es, Es.shape)
        assert indices.shape == (Es.shape[0], dim)

def test_manybody_orbitals():
    n, dim = 7, 3
    Ecut = 2
    manybody_indices, manybody_Es = manybody_orbitals(n, dim, Ecut)
    print("manybody_indices:\n", manybody_indices, manybody_indices.shape)
    print("manybody_Es:\n", manybody_Es, manybody_Es.shape)
    assert manybody_indices.shape == (manybody_Es.shape[0], n, dim)
