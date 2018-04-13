import numpy as np
from numba import jit, float64, int64
from numba.types import Tuple


class _integrals:
    
    def __init__(self, molecule_object):
        self.molecule = molecule_object

        
    def Overlap_integral(self, bf1, bf2):
        None
        
    def Kinetic_energy_integral(self, bf1, bf2):
        None
        
    def Nuclear_electron_attraction_integral(self, bf1, bf2):
        None
        
    def Electron_electron_repulsion_integral(self, bf1, bf2, bf3, bf4):
        None
        
    def Nuclear_nuclear_repulsin(self):
        None
        
    def Multipole_moment_integral(self, bf1, bf2):
        None
