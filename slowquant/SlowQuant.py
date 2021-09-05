import numpy as np
from slowquant.molecule.moleculeclass import _Molecule
from slowquant.molecularintegrals.integralclass import _Integrals
#from slowquant.hartreefock.hartreefockclass import _HartreeFock

class SlowQuant:
    """
    Main class of SlowQuant
    
    To use:
    
    set_molecule("molecule_file"), only .xyz
    Molecule.set_basis_set("basis_set_name")
    """
    def __init__(self):
        self.Molecule = False
        
    def set_molecule(self, molecule_file, set_unit="angstrom"):
        """
        Loads in the molecule information
        
        Input : molecule_file, can only be xyz format.
        Optional input : set_unit {bohr, angstrom(Default)}
        """
        self.Molecule = _Molecule(molecule_file, unit=set_unit)
        
    def initialize_integrals(self):
        self.Integrals = _Integrals(self.Molecule)
        
    def hartree_fock(self):
        self.hartree_fock = _HartreeFock(self.molecule, self.Integrals)
         