import numpy as np
from slowquant import SlowQuant as sq


def test_integrals_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("DZ")
    A.initialize_integrals()
    
    S_check = np.load("data/testfiles/S_STO2G_Li2.npy")
    V_check = np.load("data/testfiles/V_STO2G_Li2.npy")
    #T_check = np.load("data/testfiles/T_STO2G_Li2.npy")
    ERI_check = np.load("data/testfiles/ERI_STO2G_Li2.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix()))
    assert np.max(np.abs(V_check - A.Integrals.get_Nuclear_electron_matrix()))
    #assert np.max(np.abs(T_check - A.Integrals.get_Kinetic_energy_matrix()))
    assert np.max(np.abs(ERI_check - A.Integrals.get_Electron_electron_matrix()))