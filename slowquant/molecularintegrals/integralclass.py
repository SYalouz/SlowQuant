import numpy as np
from numba import jit, float64, int64
from slowquant.molecularintegrals.overlap import *
from slowquant.molecularintegrals.nuclear_electron_potential import *
from slowquant.molecularintegrals.electron_electron import *


class _Integrals:
    def __init__(self, molecule_object):
        self.molecule_obj = molecule_object
        self._output_buffer = np.zeros((81)) # Up to p functions
        self._primitives_buffer = np.zeros((10,10,9)) # Up to p functions, and up to 10 primitives
        self._primitives_buffer_2e = np.zeros((10,10,10,10,81)) # Up to p functions, and up to 10 primitives
        self._Contraction_1_buffer = np.zeros(10) # Up to 10 primitives
        self._Contraction_2_buffer = np.zeros(10) # Up to 10 primitives
        self._Contraction_3_buffer = np.zeros(10) # Up to 10 primitives
        self._Contraction_4_buffer = np.zeros(10) # Up to 10 primitives
        self._primitives_buffer_2 = np.zeros(81) # Up to p functions
        self._E_buffer = np.zeros((3,3,3,3)) # Up to p functions
        self._E_buffer_2 = np.zeros((3,3,3,3)) # Up to p functions
        self._R_buffer = np.zeros((5,5,5,5)) # Up to p functions
        
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
        
    def Nuclear_electron_attraction_integral(self, shell_number_1, shell_number_2):
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
            return nuclear_electron_integral_0_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer, self.molecule_obj.get_molecule_charge_xyz(), self._E_buffer, self._R_buffer, self._primitives_buffer_2[:1])
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return nuclear_electron_integral_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer, self.molecule_obj.get_molecule_charge_xyz(), self._E_buffer, self._R_buffer, self._primitives_buffer_2[:3])
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return nuclear_electron_integral_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._Contraction_1_buffer, self._Contraction_2_buffer, self.molecule_obj.get_molecule_charge_xyz(), self._E_buffer, self._R_buffer, self._primitives_buffer_2[:9])
        
    def Electron_electron_repulsion_integral(self, shell_number_1, shell_number_2, shell_number_3, shell_number_4):
        angular_moment_1 = self.molecule_obj._basis_shell_list[shell_number_1].angular_moment
        angular_moment_2 = self.molecule_obj._basis_shell_list[shell_number_2].angular_moment
        angular_moment_3 = self.molecule_obj._basis_shell_list[shell_number_3].angular_moment
        angular_moment_4 = self.molecule_obj._basis_shell_list[shell_number_4].angular_moment
        if angular_moment_1 < angular_moment_2:
            shell_number_1, shell_number_2 = shell_number_2, shell_number_1
            angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        if angular_moment_3 < angular_moment_4:
            shell_number_3, shell_number_4 = shell_number_4, shell_number_3
            angular_moment_3, angular_moment_4 = angular_moment_4, angular_moment_3
        if angular_moment_1*(angular_moment_1+1)//2+angular_moment_2 < angular_moment_3*(angular_moment_3+1)//2+angular_moment_4:
            shell_number_1, shell_number_2, shell_number_3, shell_number_4 = shell_number_3, shell_number_4, shell_number_1, shell_number_2
            angular_moment_1, angular_moment_2, angular_moment_3, angular_moment_4 = angular_moment_3, angular_moment_4, angular_moment_1, angular_moment_2
        Coord_1 = self.molecule_obj._basis_shell_list[shell_number_1].coord
        Coord_2 = self.molecule_obj._basis_shell_list[shell_number_2].coord
        Coord_3 = self.molecule_obj._basis_shell_list[shell_number_3].coord
        Coord_4 = self.molecule_obj._basis_shell_list[shell_number_4].coord
        gauss_exp_1 = self.molecule_obj._basis_shell_list[shell_number_1].exponents
        gauss_exp_2 = self.molecule_obj._basis_shell_list[shell_number_2].exponents
        gauss_exp_3 = self.molecule_obj._basis_shell_list[shell_number_3].exponents
        gauss_exp_4 = self.molecule_obj._basis_shell_list[shell_number_4].exponents
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].contraction_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].contraction_coeffs
        Contra_coeffs_3 = self.molecule_obj._basis_shell_list[shell_number_3].contraction_coeffs
        Contra_coeffs_4 = self.molecule_obj._basis_shell_list[shell_number_4].contraction_coeffs
        if angular_moment_1 == 0 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_0_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:1], self._output_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:3], self._output_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:9], self._output_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:9], self._output_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:27], self._output_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 1:
            return electron_electron_integral_1_1_1_1(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._Contraction_1_buffer, self._Contraction_2_buffer, self._Contraction_3_buffer, self._Contraction_4_buffer, self._E_buffer, self._E_buffer_2, self._R_buffer, self._primitives_buffer_2[:81], self._output_buffer)
        
    def Nuclear_nuclear_repulsion(self):
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
        
    def get_idx_list_two_electron(self, shell_number_1, shell_number_2, shell_number_3, shell_number_4):
        angular_moment_1 = self.molecule_obj._basis_shell_list[shell_number_1].angular_moment
        angular_moment_2 = self.molecule_obj._basis_shell_list[shell_number_2].angular_moment
        angular_moment_3 = self.molecule_obj._basis_shell_list[shell_number_3].angular_moment
        angular_moment_4 = self.molecule_obj._basis_shell_list[shell_number_4].angular_moment
        if angular_moment_1 < angular_moment_2:
            shell_number_1, shell_number_2 = shell_number_2, shell_number_1
            angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        if angular_moment_3 < angular_moment_4:
            shell_number_3, shell_number_4 = shell_number_4, shell_number_3
            angular_moment_3, angular_moment_4 = angular_moment_4, angular_moment_3
        if angular_moment_1*(angular_moment_1+1)//2+angular_moment_2 < angular_moment_3*(angular_moment_3+1)//2+angular_moment_4:
            shell_number_1, shell_number_2, shell_number_3, shell_number_4 = shell_number_3, shell_number_4, shell_number_1, shell_number_2
            angular_moment_1, angular_moment_2, angular_moment_3, angular_moment_4 = angular_moment_3, angular_moment_4, angular_moment_1, angular_moment_2
        idx_1 = self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx
        idx_2 = self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx
        idx_3 = self.molecule_obj._basis_shell_list[shell_number_3].basis_function_idx
        idx_4 = self.molecule_obj._basis_shell_list[shell_number_4].basis_function_idx
        output = []
        for i in range(0, len(idx_1)):
            for j in range(0, len(idx_2)):
                for k in range(0, len(idx_3)):
                    for l in range(0, len(idx_4)):
                        output.append([idx_1[i],idx_2[j],idx_3[k],idx_4[l]])
        return output
        
    def get_Overlap_matrix(self):
        Overlap_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                temp = self.Overlap_integral(i, j)
                idx = self.get_idx_list_one_electron(i, j)
                for k in range(0, len(idx)):
                    Overlap_matrix[idx[k][0],idx[k][1]] = Overlap_matrix[idx[k][1],idx[k][0]] = temp[k]
        return Overlap_matrix
        
    def get_Nuclear_electron_matrix(self):
        Nuclear_electron_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                temp = self.Nuclear_electron_attraction_integral(i, j)
                idx = self.get_idx_list_one_electron(i, j)
                for k in range(0, len(idx)):
                    Nuclear_electron_matrix[idx[k][0],idx[k][1]] = Nuclear_electron_matrix[idx[k][1],idx[k][0]] = temp[k]
        return Nuclear_electron_matrix
        
    def get_Electron_electron_matrix(self):
        Electron_electron_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function(),self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                for k in range(0, self.molecule_obj.get_number_shells()):
                    for l in range(k, self.molecule_obj.get_number_shells()):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            temp = self.Electron_electron_repulsion_integral(i,j,k,l)
                            idx = self.get_idx_list_two_electron(i,j,k,l)
                            for m in range(0, len(idx)):
                                Electron_electron_matrix[idx[m][0],idx[m][1],idx[m][2],idx[m][3]] = Electron_electron_matrix[idx[m][1],idx[m][0],idx[m][2],idx[m][3]] = Electron_electron_matrix[idx[m][0],idx[m][1],idx[m][3],idx[m][2]] =Electron_electron_matrix[idx[m][1],idx[m][0],idx[m][3],idx[m][2]] = Electron_electron_matrix[idx[m][2],idx[m][3],idx[m][1],idx[m][0]] = Electron_electron_matrix[idx[m][2],idx[m][3],idx[m][0],idx[m][1]] = Electron_electron_matrix[idx[m][3],idx[m][2],idx[m][1],idx[m][0]] = Electron_electron_matrix[idx[m][3],idx[m][2],idx[m][0],idx[m][1]] = temp[m]
        return Electron_electron_matrix















