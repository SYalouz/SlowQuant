import numpy as np
from slowquant.molecule.moleculeclass import _Molecule


class SlowQuant:
    def __init__(self):
        self.Molecule = False
        self.a = 1
        
    def set_molecule(self, molecule_file):
        self.Molecule = _Molecule(molecule_file)