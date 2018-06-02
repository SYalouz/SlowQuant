import numpy as np
from numba import jit, float64
from numba.types import Tuple
from slowquant.molecularintegrals.utility import Contraction_two_electron
from math import erf
from numpy import exp

@jit(Tuple((float64[:,:],float64[:,:,:]))(float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:,:]), nopython=True, cache=True)
def bra_expansion_coeffs_0_0_handtuned(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    XAB_left2 = (Coord_1 - Coord_2)**2.0
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = bra_array[0,i,j] = gauss_exp_1_left + gauss_exp_2_left
            q_left = bra_array[1,i,j] = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = bra_array[2:5,i,j] = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            E1 = exp(-q_left*XAB_left2)
            E_buff_1[i,j] = E1[0]*E1[1]*E1[2]
    return E_buff_1, bra_array


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:,:], float64[:,:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_0_0_0_0_handtuned(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, E_buff_1, E_buff_2, output_buffer, Norm_array, bra_array, ket_array):
    number_primitive_1 = Contra_coeffs_1.shape[0]
    number_primitive_2 = Contra_coeffs_2.shape[0]
    number_primitive_3 = Contra_coeffs_3.shape[0]
    number_primitive_4 = Contra_coeffs_4.shape[0]
    XAB_right2 = (Coord_3 - Coord_4)**2.0
    for k in range(0, number_primitive_3):
        gauss_exp_1_right = gauss_exp_3[k]
        for l in range(0, number_primitive_4):
            gauss_exp_2_right = gauss_exp_4[l]
            p_right = ket_array[0,k,l] = gauss_exp_1_right + gauss_exp_2_right
            q_right = ket_array[1,k,l] = gauss_exp_1_right * gauss_exp_2_right / p_right
            P_right = ket_array[2:5,k,l] = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
            E2 = exp(-q_right*XAB_right2)
            E_buff_2[k,l] = E2[0]*E2[1]*E2[2] 
    
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            p_left = bra_array[0,i,j]
            q_left = bra_array[1,i,j]
            P_left = bra_array[2:5,i,j]
            E1prod = E_buff_1[i,j]
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    p_right = ket_array[0,k,l]
                    q_right = ket_array[1,k,l]
                    P_right = ket_array[2:5,k,l]
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC2 = (XPC)**2+(YPC)**2+(ZPC)**2
                    E2prod = E_buff_2[k,l]
                    z = alpha*RPC2
                    if z < 10**-10:
                        R = 1.0
                    else:
                        R = (3.141592653589793238462643383279/(4.0*z))**0.5*erf(z**0.5)

                    primitives_buffer[i,j,k,l,0] = R*E1prod*E2prod/(p_left*p_right*(p_left+p_right)**0.5)

    output_buffer[0] = Contraction_two_electron(primitives_buffer[:,:,:,:,0], Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4)*9.027033336764104
    return output_buffer