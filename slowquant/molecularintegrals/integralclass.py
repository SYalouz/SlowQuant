import numpy as np
import math
from numba import jit, float64, int64
from slowquant.molecularintegrals.overlap_MD import *
from slowquant.molecularintegrals.nuclear_electron_MD2 import *
from slowquant.molecularintegrals.electron_electron_MD4 import *
from slowquant.molecularintegrals.utility import put_in_array_ERI, make_idx_list_two_electron, make_idx_list_one_electron, PsuedoNorm, PsuedoNorm2


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
        self._idx_buffer_1e = np.zeros((9,2),dtype=int) # Up to p functions
        self._idx_buffer_2e = np.zeros((81,4),dtype=int) # Up to p functions
        self._norm_array = np.zeros((3,3,3)) # Up to p functions
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    self._norm_array[i,j,k] = PsuedoNorm2(i, j, k)
        for i in range(0, self.molecule_obj.get_number_shells()):
            temp = np.zeros(len(self.molecule_obj._basis_shell_list[i].contraction_coeffs))
            for j in range(0, len(self.molecule_obj._basis_shell_list[i].contraction_coeffs)):
                temp[j] = PsuedoNorm(self.molecule_obj._basis_shell_list[i].angular_moment, self.molecule_obj._basis_shell_list[i].exponents[j])*self.molecule_obj._basis_shell_list[i].contraction_coeffs[j]
            self.molecule_obj._basis_shell_list[i].pseudo_normed_contract_coeffs = temp
        
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
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].pseudo_normed_contract_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].pseudo_normed_contract_coeffs
        if angular_moment_1 == 0 and angular_moment_2 == 0:
            return overlap_integral_0_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return overlap_integral_1_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return overlap_integral_1_1_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer)
        
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
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].pseudo_normed_contract_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].pseudo_normed_contract_coeffs
        if angular_moment_1 == 0 and angular_moment_2 == 0:
            return nuclear_electron_integral_0_0_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer, self._R_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return nuclear_electron_integral_1_0_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer, self._R_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return nuclear_electron_integral_1_1_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer, self._R_buffer)
        
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
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].pseudo_normed_contract_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].pseudo_normed_contract_coeffs
        Contra_coeffs_3 = self.molecule_obj._basis_shell_list[shell_number_3].pseudo_normed_contract_coeffs
        Contra_coeffs_4 = self.molecule_obj._basis_shell_list[shell_number_4].pseudo_normed_contract_coeffs
        if angular_moment_1 == 0 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_0_0_0_0_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_0_0_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_0_0_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_1_0_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_1_0_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 1:
            return electron_electron_integral_1_1_1_1_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._R_buffer, self._output_buffer, self._norm_array)
        
    def Nuclear_nuclear_repulsion(self):
        None
        
    def get_idx_list_one_electron(self, shell_number_1, shell_number_2):
        return make_idx_list_one_electron(self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx, self._idx_buffer_1e)
        
    def get_idx_list_two_electron(self, shell_number_1, shell_number_2, shell_number_3, shell_number_4):
        return make_idx_list_two_electron(self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_3].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_4].basis_function_idx,self._idx_buffer_2e)
        
    def generate_shell_number_two_electron(self):
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                for k in range(0, self.molecule_obj.get_number_shells()):
                    for l in range(k, self.molecule_obj.get_number_shells()):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            yield i, j, k, l

    def get_Overlap_matrix(self):
        Overlap_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                temp = self.Overlap_integral(i, j)
                idx = self.get_idx_list_one_electron(i, j)
                for k in range(0, len(idx)):
                    Overlap_matrix[idx[k,0],idx[k,1]] = Overlap_matrix[idx[k,1],idx[k,0]] = temp[k]
        return Overlap_matrix
        
    def get_Nuclear_electron_matrix(self):
        Nuclear_electron_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                temp = self.Nuclear_electron_attraction_integral(i, j)
                idx = self.get_idx_list_one_electron(i, j)
                for k in range(0, len(idx)):
                    Nuclear_electron_matrix[idx[k,0],idx[k,1]] = Nuclear_electron_matrix[idx[k,1],idx[k,0]] = temp[k]
        return Nuclear_electron_matrix
        
    def get_Electron_electron_matrix(self):
        Electron_electron_matrix = np.zeros((self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function(),self.molecule_obj.get_number_basis_function(), self.molecule_obj.get_number_basis_function()))
        for i in range(0, self.molecule_obj.get_number_shells()):
            for j in range(i, self.molecule_obj.get_number_shells()):
                for k in range(0, self.molecule_obj.get_number_shells()):
                    for l in range(k, self.molecule_obj.get_number_shells()):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            temp = self.Electron_electron_repulsion_integral(i,j,k,l)
                            Electron_electron_matrix = put_in_array_ERI(self.molecule_obj._basis_shell_list[i].basis_function_idx, self.molecule_obj._basis_shell_list[j].basis_function_idx, self.molecule_obj._basis_shell_list[k].basis_function_idx, self.molecule_obj._basis_shell_list[l].basis_function_idx,Electron_electron_matrix,temp)
        return Electron_electron_matrix
















