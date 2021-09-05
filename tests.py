import numpy as np
import data.generator.generate_expansion_coefficients as genExpCoeff
import data.generator.generate_hermite_integral as genHerInt
import data.generator.generate_electron_electron_MD4 as genERI
import data.generator.generate_nuclear_electron_MD2 as genNucEl
import data.generator.generate_overlap_MD as genOver
import data.generator.generate_bra_expansion_coeffs as genbraexp
import data.generator.generate_kinetic_energy_MD as genKE
max_angular = 1
genExpCoeff.generate_expansion_coefficients(max_angular)
genHerInt.write_hermite_integral(max_angular)
genERI.write_electron_electron(max_angular)
genNucEl.write_nuclear_electron(max_angular)
genOver.write_overlap(max_angular)
genbraexp.write_bra_expansion_coeffs(max_angular)
<<<<<<< HEAD
#genKE.write_kinetic_energy(max_angular)
=======
genKE.write_kinetic_energy(max_angular)
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
from slowquant import SlowQuant as sq

def test_overlap_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li4.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
<<<<<<< HEAD
    S_check = np.load("data/testfiles/S_STO2G_Li4.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix())) < 10**-8
=======
    S_reference = np.load("data/testfiles/S_STO2G_Li4.npy")
    S = A.Integrals.get_Overlap_matrix()
    
    np.testing.assert_allclose(S, S_reference, atol=1e-12)
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9


def test_nuclear_electron_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li4.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
<<<<<<< HEAD
    V_check = np.load("data/testfiles/V_STO2G_Li4.npy")
    
    assert np.max(np.abs(V_check - A.Integrals.get_Nuclear_electron_matrix())) < 10**-8
=======
    V_reference = np.load("data/testfiles/V_STO2G_Li4.npy")
    V = A.Integrals.get_Nuclear_electron_matrix()
    
    np.testing.assert_allclose(V, V_reference, atol=1e-12)
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9

    
def test_electron_electron_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li4.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
<<<<<<< HEAD
    ERI_check = np.load("data/testfiles/ERI_STO2G_Li4.npy")
    
    assert np.max(np.abs(ERI_check - A.Integrals.get_Electron_electron_matrix())) < 10**-8
=======
    ERI_reference = np.load("data/testfiles/ERI_STO2G_Li4.npy")
    ERI = A.Integrals.get_Electron_electron_matrix()
    
    np.testing.assert_allclose(ERI, ERI_reference, atol=1e-12)
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
    
    
def test_kinetic_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li4.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
<<<<<<< HEAD
    T_check = np.load("data/testfiles/T_STO2G_Li4.npy")
    
    assert np.max(np.abs(T_check - A.Integrals.get_Kinetic_energy_matrix())) < 10**-8
=======
    T_reference = np.load("data/testfiles/T_STO2G_Li4.npy")
    T = A.Integrals.get_Kinetic_energy_matrix()
    
    np.testing.assert_allclose(T, T_reference, atol=1e-12)
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9

"""
def test_overlap_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
<<<<<<< HEAD
    S_check = np.load("data/testfiles/S_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix())) < 10**-8
=======
    S_reference = np.load("data/testfiles/S_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(S_reference - A.Integrals.get_Overlap_matrix())) < 10**-8
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9


def test_nuclear_electron_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
<<<<<<< HEAD
    V_check = np.load("data/testfiles/V_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(V_check - A.Integrals.get_Nuclear_electron_matrix())) < 10**-8
=======
    V_reference = np.load("data/testfiles/V_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(V_reference - A.Integrals.get_Nuclear_electron_matrix())) < 10**-8
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9

    
def test_electron_electron_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
<<<<<<< HEAD
    ERI_check = np.load("data/testfiles/ERI_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(ERI_check - A.Integrals.get_Electron_electron_matrix())) < 10**-8
=======
    ERI_reference = np.load("data/testfiles/ERI_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(ERI_reference - A.Integrals.get_Electron_electron_matrix())) < 10**-8
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
"""    
"""
def test_kinetic_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
<<<<<<< HEAD
    T_check = np.load("data/testfiles/T_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(T_check - A.Integrals.Kinetic_energy_integral())) < 10**-8
=======
    T_reference = np.load("data/testfiles/T_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(T_reference - A.Integrals.Kinetic_energy_integral())) < 10**-8
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
"""
