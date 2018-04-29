import numpy as np
from numba import jit, float64, int64
from slowquant.molecularintegrals.overlap import *


class _Integrals:
    def __init__(self, molecule_object):
        self.molecule_obj = molecule_object
        self._output_buffer = np.zeros((9)) # Up to p functions
        self._primitives_buffer = np.zeros((10,10,9)) # Up to p functions, and up to 10 primitives
        self._Contraction_1_buffer = np.zeros(10) # Up to 10 primitives
        self._Contraction_2_buffer = np.zeros(10) # Up to 10 primitives
        
    def Overlap_integral(self, shell_number_1, shell_number_2):
        angular_moment_1 = self.molecule_obj._basis_shell_list[shell_number_1].angular_moment
        angular_moment_2 = self.molecule_obj._basis_shell_list[shell_number_2].angular_moment
        if angular_moment_1 < angular_moment_2:
            shell_number_1, shell_number_2 = shell_number_2, shell_number_1
            angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        Coord_1 = self.molecule_obj._basis_shell_list[shell_number_1].coord
        Coord_2 = self.molecule_obj._basis_shell_list[shell_number_2].coord
        gauss_exp_1 = self.molecule_obj._basis_shell_list[shell_number_1].exponents
        gauss_exp_2 = self.molecule_obj._basis_shell_list[shell_number_2].exponents
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].contraction_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].contraction_coeffs
        if angular_moment_1 == 0 and angular_moment_2 == 0:
            return overlap_integral_0_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return overlap_integral_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return overlap_integral_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer)
        
    def Kinetic_energy_integral(self, shell1, shell2):
        None
        
    def Nuclear_electron_attraction_integral(self, shell1, shell2):
        None
        
    def Electron_electron_repulsion_integral(self, shell1, shell2, shell3, shell4):
        None
        
    def Nuclear_nuclear_repulsion(self):
        None
        
    def Multipole_moment_integral(self, shell1, shell2):
        None
        
    def get_idx_list_one_electron(self, shell_number_1, shell_number_2):
        angular_moment_1 = self.molecule_obj._basis_shell_list[shell_number_1].angular_moment
        angular_moment_2 = self.molecule_obj._basis_shell_list[shell_number_2].angular_moment
        if angular_moment_1 < angular_moment_2:
            shell_number_1, shell_number_2 = shell_number_2, shell_number_1
            angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        idx_1 = self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx
        idx_2 = self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx
        if angular_moment_1 == 0:
            output = [[idx_1[0], idx_2[0]]]
        elif angular_moment_1 == 1:
            if angular_moment_2 == 0:
                output = [[idx_1[0], idx_2[0]],
                          [idx_1[1], idx_2[0]],
                          [idx_1[2], idx_2[0]]]
            else: # angular_moment_2 == 1
                output = [[idx_1[0], idx_2[0]],
                          [idx_1[0], idx_2[1]],
                          [idx_1[0], idx_2[2]],
                          [idx_1[1], idx_2[0]],
                          [idx_1[1], idx_2[1]],
                          [idx_1[1], idx_2[2]],
                          [idx_1[2], idx_2[0]],
                          [idx_1[2], idx_2[1]],
                          [idx_1[2], idx_2[2]]]
        return output
        
    def get_Overlap_matrix(self):
        Overlap_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                temp = self.Overlap_integral(i, j)
                idx = self.get_idx_list_one_electron(i, j)
                print(temp)
                print(idx)
                print(" ")
                for k in range(0, len(idx)):
                    Overlap_matrix[idx[k][0],idx[k][1]] = Overlap_matrix[idx[k][1],idx[k][0]] = temp[k]
        return Overlap_matrix
















