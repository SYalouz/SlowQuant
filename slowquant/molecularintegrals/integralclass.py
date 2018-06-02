import numpy as np
import math
from numba import jit, float64, int64
from slowquant.molecularintegrals.overlap_MD import *
from slowquant.molecularintegrals.nuclear_electron_MD2 import *
from slowquant.molecularintegrals.electron_electron_MD4 import *
from slowquant.molecularintegrals.handtuned_integrals import *
from slowquant.molecularintegrals.bra_expansion_coeffs import *
from slowquant.molecularintegrals.utility import put_in_array_ERI, make_idx_list_two_electron, make_idx_list_one_electron, PsuedoNorm, PsuedoNorm2


class _Integrals:
    def __init__(self, molecule_object):
        max_angular = 2
        max_comb    = (max_angular+1)*(max_angular+1+1)//2 
        max_prim    = 10
        self.molecule_obj = molecule_object
        self._output_buffer = np.zeros((max_comb**4)) 
        self._primitives_buffer = np.zeros((max_prim,max_prim,max_comb**2))
        self._primitives_buffer_2e = np.zeros((max_prim,max_prim,max_prim,max_prim,max_comb**4)) 
        self._E_buffer_1_e = np.zeros((max_comb,max_comb,max_comb,3))
        self._E_buffer = np.zeros((max_prim,max_prim,max_comb,max_comb,max_comb,3)) 
        self._E_buffer_2 = np.zeros((max_prim,max_prim,max_comb,max_comb,max_comb,3))
        self._E_buffer_ssss_1 = np.zeros((max_prim,max_prim)) # Special case for ssss integrals
        self._E_buffer_ssss_2 = np.zeros((max_prim,max_prim)) # Special case for ssss integrals
        self._R_buffer = np.zeros((max_angular*4+1,max_angular*4+1,max_angular*4+1,max_angular*4+1)) 
        self._idx_buffer_1e = np.zeros((max_comb**2,2),dtype=int) 
        self._idx_buffer_2e = np.zeros((max_comb**4,4),dtype=int) 
        self._norm_array = np.zeros((max_angular+1,max_angular+1,max_angular+1)) 
        self._ket_precomputed = np.zeros((5,max_prim,max_prim)) #  p_right, q_right, P_right_x, P_right_y, P_right_z
        self._bra_precomputed = np.zeros((5,max_prim,max_prim)) #  p_right, q_right, P_right_x, P_right_y, P_right_z
        self._latest_idx_1 = -1
        self._latest_idx_2 = -1
        for i in range(0, max_angular+1):
            for j in range(0, max_angular+1):
                for k in range(0, max_angular+1):
                    self._norm_array[i,j,k] = PsuedoNorm2(i, j, k)
        for i in range(0, self.molecule_obj.get_number_shells()):
            temp = np.zeros(len(self.molecule_obj._basis_shell_list[i].contraction_coeffs))
            for j in range(0, len(self.molecule_obj._basis_shell_list[i].contraction_coeffs)):
                temp[j] = PsuedoNorm(self.molecule_obj._basis_shell_list[i].angular_moment, self.molecule_obj._basis_shell_list[i].exponents[j])*self.molecule_obj._basis_shell_list[i].contraction_coeffs[j]
            self.molecule_obj._basis_shell_list[i].pseudo_normed_contract_coeffs = temp
            
        self._s_functions = []
        self._p_functions = []
        for i in range(self.molecule_obj.get_number_shells()):
            if self.molecule_obj._basis_shell_list[i].angular_moment == 0:
                self._s_functions.append(i)
            elif self.molecule_obj._basis_shell_list[i].angular_moment == 1:
                self._p_functions.append(i)
        self._s_functions = np.array(self._s_functions)
        self._p_functions = np.array(self._p_functions)
        
        
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
            return overlap_integral_0_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e)
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return overlap_integral_1_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e)
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return overlap_integral_1_1_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e)

        
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
            return nuclear_electron_integral_0_0_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e, self._R_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 0:
            return nuclear_electron_integral_1_0_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e, self._R_buffer)
        elif angular_moment_1 == 1 and angular_moment_2 == 1:
            return nuclear_electron_integral_1_1_MD2(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, self.molecule_obj.get_molecule_charge_xyz(), self._output_buffer, self._primitives_buffer, self._norm_array, self._E_buffer_1_e, self._R_buffer)

        
    def Electron_electron_repulsion_integral(self, shell_number_1, shell_number_2, shell_number_3, shell_number_4):
        angular_moment_1 = self.molecule_obj._basis_shell_list[shell_number_1].angular_moment
        angular_moment_2 = self.molecule_obj._basis_shell_list[shell_number_2].angular_moment
        angular_moment_3 = self.molecule_obj._basis_shell_list[shell_number_3].angular_moment
        angular_moment_4 = self.molecule_obj._basis_shell_list[shell_number_4].angular_moment
                
        Coord_3 = self.molecule_obj._basis_shell_list[shell_number_3].coord
        Coord_4 = self.molecule_obj._basis_shell_list[shell_number_4].coord
        gauss_exp_3 = self.molecule_obj._basis_shell_list[shell_number_3].exponents
        gauss_exp_4 = self.molecule_obj._basis_shell_list[shell_number_4].exponents
        Contra_coeffs_1 = self.molecule_obj._basis_shell_list[shell_number_1].pseudo_normed_contract_coeffs
        Contra_coeffs_2 = self.molecule_obj._basis_shell_list[shell_number_2].pseudo_normed_contract_coeffs
        Contra_coeffs_3 = self.molecule_obj._basis_shell_list[shell_number_3].pseudo_normed_contract_coeffs
        Contra_coeffs_4 = self.molecule_obj._basis_shell_list[shell_number_4].pseudo_normed_contract_coeffs
        
        if self._latest_idx_1 != self.molecule_obj._basis_shell_list[shell_number_1].shell_idx or self._latest_idx_2 != self.molecule_obj._basis_shell_list[shell_number_2].shell_idx:
            self._latest_idx_1 = self.molecule_obj._basis_shell_list[shell_number_1].shell_idx
            self._latest_idx_2 = self.molecule_obj._basis_shell_list[shell_number_2].shell_idx
            Coord_1 = self.molecule_obj._basis_shell_list[shell_number_1].coord
            Coord_2 = self.molecule_obj._basis_shell_list[shell_number_2].coord
            gauss_exp_1 = self.molecule_obj._basis_shell_list[shell_number_1].exponents
            gauss_exp_2 = self.molecule_obj._basis_shell_list[shell_number_2].exponents
            if angular_moment_1 == 0 and angular_moment_2 == 0:
                self._E_buffer_ssss_1, self._bra_precomputed = bra_expansion_coeffs_0_0_handtuned(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, self._E_buffer_ssss_1, self._bra_precomputed)
            elif angular_moment_1 == 1 and angular_moment_2 == 0:
                self._E_buffer, self._bra_precomputed = bra_expansion_coeffs_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, self._E_buffer, self._bra_precomputed)
            elif angular_moment_1 == 1 and angular_moment_2 == 1:
                self._E_buffer, self._bra_precomputed = bra_expansion_coeffs_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, self._E_buffer, self._bra_precomputed)
            
        if angular_moment_1 == 0 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            Coord_1 = self.molecule_obj._basis_shell_list[shell_number_1].coord
            Coord_2 = self.molecule_obj._basis_shell_list[shell_number_2].coord
            gauss_exp_1 = self.molecule_obj._basis_shell_list[shell_number_1].exponents
            gauss_exp_2 = self.molecule_obj._basis_shell_list[shell_number_2].exponents
            return electron_electron_integral_0_0_0_0_handtuned(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer_ssss_1, self._E_buffer_ssss_2, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_0_0_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._E_buffer_2, self._R_buffer, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 0 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_0_0_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._E_buffer_2, self._R_buffer, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        elif angular_moment_1 == 1 and angular_moment_2 == 0 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_0_1_0_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._E_buffer_2, self._R_buffer, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 0:
            return electron_electron_integral_1_1_1_0_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._E_buffer_2, self._R_buffer, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        elif angular_moment_1 == 1 and angular_moment_2 == 1 and angular_moment_3 == 1 and angular_moment_4 == 1:
            return electron_electron_integral_1_1_1_1_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, self._primitives_buffer_2e, self._E_buffer, self._E_buffer_2, self._R_buffer, self._output_buffer, self._norm_array, self._bra_precomputed, self._ket_precomputed)
        else:
            print("FATAL ERROR: Wrong order in integrals, ang. moments: ", angular_moment_1, angular_moment_2, angular_moment_3, angular_moment_4)
            
            
    def Nuclear_nuclear_repulsion(self):
        None
        
        
    def get_idx_list_one_electron(self, shell_number_1, shell_number_2):
        return make_idx_list_one_electron(self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx, self._idx_buffer_1e)
        
        
    def get_idx_list_two_electron(self, shell_number_1, shell_number_2, shell_number_3, shell_number_4):
        return make_idx_list_two_electron(self.molecule_obj._basis_shell_list[shell_number_1].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_2].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_3].basis_function_idx, self.molecule_obj._basis_shell_list[shell_number_4].basis_function_idx,self._idx_buffer_2e)
        

    def generate_shell_number_two_electron(self):
        number_s = self._s_functions.shape[0]
        number_p = self._p_functions.shape[0]
        s_func = self._s_functions
        p_func = self._p_functions
        for i in range(number_s-1, -1, -1):
            for j in range(i, -1, -1): # number_s
                for k in range(number_s):
                    for l in range(k, number_s):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            yield s_func[i], s_func[j], s_func[k], s_func[l]
                            
        for i in range(number_p-1, -1, -1):
            for j in range(number_s-1, -1, -1):
                for k in range(number_s):
                    for l in range(k, number_s):
                        yield p_func[i], s_func[j], s_func[k], s_func[l]
                for k in range(number_p):
                    for l in range(number_s):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            yield p_func[i], s_func[j], p_func[k], s_func[l]
                            
        for i in range(number_p-1, -1, -1):
            for j in range(i, -1, -1): # number_p
                for k in range(number_s):
                    for l in range(k, number_s):
                        yield p_func[i], p_func[j], s_func[k], s_func[l]
                for k in range(number_p):
                    for l in range(number_s):
                        yield p_func[i], p_func[j], p_func[k], s_func[l]
                for k in range(number_p):
                    for l in range(k, number_p):
                        if i*(i+1)//2+j >= k*(k+1)//2+l:
                            yield p_func[i], p_func[j], p_func[k], p_func[l]
                            

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
        integrals = self.generate_shell_number_two_electron()
        for i,j,k,l in integrals:
            temp = self.Electron_electron_repulsion_integral(i,j,k,l)
            Electron_electron_matrix = put_in_array_ERI(self.molecule_obj._basis_shell_list[i].basis_function_idx, self.molecule_obj._basis_shell_list[j].basis_function_idx, self.molecule_obj._basis_shell_list[k].basis_function_idx, self.molecule_obj._basis_shell_list[l].basis_function_idx,Electron_electron_matrix,temp)
        return Electron_electron_matrix
















