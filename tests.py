import numpy as np
from slowquant import SlowQuant as sq


def test_overlap_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
    S_check = np.load("data/testfiles/S_STO2G_Li2.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix())) < 10**-8


def test_nuclear_electron_up_to_p():
    None
    
    
def test_electron_electron_up_to_p():
    None