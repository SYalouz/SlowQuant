import numpy as np
import data.generator.generate_expansion_coefficients as genExpCoeff
import data.generator.generate_hermite_integral as genHerInt
import data.generator.generate_electron_electron_MD4 as genERI
import data.generator.generate_nuclear_electron_MD2 as genNucEl
import data.generator.generate_overlap_MD as genOver
max_angular = 1
genExpCoeff.generate_expansion_coefficients(max_angular)
genHerInt.write_hermite_integral(max_angular)
genERI.write_electron_electron(max_angular)
genNucEl.write_nuclear_electron(max_angular)
genOver.write_overlap(max_angular)
from slowquant import SlowQuant as sq

def test_overlap_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
    S_check = np.load("data/testfiles/S_STO2G_Li2.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix())) < 10**-8


def test_nuclear_electron_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
    V_check = np.load("data/testfiles/V_STO2G_Li2.npy")
    
    assert np.max(np.abs(V_check - A.Integrals.get_Nuclear_electron_matrix())) < 10**-8

    
def test_electron_electron_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
    ERI_check = np.load("data/testfiles/ERI_STO2G_Li2.npy")
    
    assert np.max(np.abs(ERI_check - A.Integrals.get_Electron_electron_matrix())) < 10**-8
    
"""
def test_kinetic_up_to_p():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("STO2G")
    A.initialize_integrals()
    
    T_check = np.load("data/testfiles/T_STO2G_Li2.npy")
    
    assert np.max(np.abs(T_check - A.Integrals.Kinetic_energy_integral())) < 10**-8
"""
"""
def test_overlap_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
    S_check = np.load("data/testfiles/S_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(S_check - A.Integrals.get_Overlap_matrix())) < 10**-8


def test_nuclear_electron_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
    V_check = np.load("data/testfiles/V_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(V_check - A.Integrals.get_Nuclear_electron_matrix())) < 10**-8

    
def test_electron_electron_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
    ERI_check = np.load("data/testfiles/ERI_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(ERI_check - A.Integrals.get_Electron_electron_matrix())) < 10**-8
"""    
"""
def test_kinetic_up_to_d():
    A = sq.SlowQuant()
    A.set_molecule("data/testfiles/Li2.xyz",set_unit="bohr")
    A.Molecule.set_basis_set("6-31Gs")
    A.initialize_integrals()
    
    T_check = np.load("data/testfiles/T_6-31Gs_Li2.npy")
    
    assert np.max(np.abs(T_check - A.Integrals.Kinetic_energy_integral())) < 10**-8
"""
