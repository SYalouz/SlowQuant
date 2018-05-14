import numpy as np
import math
from numba import jit, float64
from slowquant.molecularintegrals.utility import boys_function, ERI_expansion_coeff_sum


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_0_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            E1 = np.exp(-q_left*XAB_left[0]*XAB_left[0])*np.exp(-q_left*XAB_left[10]*XAB_left[1])*np.exp(-q_left*XAB_left[2]*XAB_left[2])

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5

                    if RPC == 0:
                        primitives_buffer[i,j,k,l,0] = pi52/(p_left*p_right*(p_left+p_right)**0.5)
                    else:
                        primitives_buffer[i,j,k,l,0] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*E1*np.exp(-q_right*XAB_right[0]*XAB_right[0])*np.exp(-q_right*XAB_right[1]*XAB_right[1])*np.exp(-q_right*XAB_right[2]*XAB_right[2])*(pi/(4*alpha*RPC*RPC))**0.5*math.erf(alpha*RPC*RPC)

    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    output_buffer[0] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,0]
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_1_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
            E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    XPA_right = P_right - Coord_3
                    XPB_right = P_right - Coord_4
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
                    
                    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)

                    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
                    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
                    R[0,0,1,0] = ZPC * R[0,0,0,1]
                    R[0,1,0,0] = YPC * R[0,0,0,1]
                    R[1,0,0,0] = XPC * R[0,0,0,1]

                    primitives_buffer[i,j,k,l,0] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,1,1,1)*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,1] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,1,1,1)*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,2] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,1,1,1)*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,0,0]*norm_array[0,0,0]

                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]

    for out in range(0, 3):
        for i in range(0, number_primitive_1):
            for j in range(0, number_primitive_2):
                for k in range(0, number_primitive_3):
                    for l in range(0, number_primitive_4):
                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_1_0_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
            E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    XPA_right = P_right - Coord_3
                    XPB_right = P_right - Coord_4
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
                    
                    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
                    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
                    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]

                    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
                    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
                    R[0,0,1,0] = ZPC * R[0,0,0,1]
                    R[0,0,0,2] = (-2.0*alpha)**2 * boys_function(2,alpha*RPC*RPC)
                    R[0,0,1,1] = ZPC * R[0,0,0,2]
                    R[0,0,2,0] = 1 * R[0,0,0,1] + ZPC * R[0,0,1,1]
                    R[0,1,0,0] = YPC * R[0,0,0,1]
                    R[0,1,1,0] = YPC * R[0,0,1,1]
                    R[0,1,0,1] = YPC * R[0,0,0,2]
                    R[0,2,0,0] = 1 * R[0,0,0,1] + YPC * R[0,1,0,1]
                    R[1,0,0,0] = XPC * R[0,0,0,1]
                    R[1,0,1,0] = XPC * R[0,0,1,1]
                    R[1,1,0,0] = XPC * R[0,1,0,1]
                    R[1,0,0,1] = XPC * R[0,0,0,2]
                    R[2,0,0,0] = 1 * R[0,0,0,1] + XPC * R[1,0,0,1]

                    primitives_buffer[i,j,k,l,0] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,2,1,1)*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,1] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,1,2,1)*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,2] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,1,1,1,2)*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,3] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,2,1,1)*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,4] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,1,2,1)*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,5] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,1,1,1,2)*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,6] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,2,1,1)*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,7] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,1,2,1)*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,8] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,2,1,1,2)*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,0,1]*norm_array[0,0,0]

                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]

    for out in range(0, 9):
        for i in range(0, number_primitive_1):
            for j in range(0, number_primitive_2):
                for k in range(0, number_primitive_3):
                    for l in range(0, number_primitive_4):
                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_1_1_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
            E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
            E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
            E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
            E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
            E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    XPA_right = P_right - Coord_3
                    XPB_right = P_right - Coord_4
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
                    
                    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)

                    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
                    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
                    R[0,0,1,0] = ZPC * R[0,0,0,1]
                    R[0,0,0,2] = (-2.0*alpha)**2 * boys_function(2,alpha*RPC*RPC)
                    R[0,0,1,1] = ZPC * R[0,0,0,2]
                    R[0,0,2,0] = 1 * R[0,0,0,1] + ZPC * R[0,0,1,1]
                    R[0,1,0,0] = YPC * R[0,0,0,1]
                    R[0,1,1,0] = YPC * R[0,0,1,1]
                    R[0,1,0,1] = YPC * R[0,0,0,2]
                    R[0,2,0,0] = 1 * R[0,0,0,1] + YPC * R[0,1,0,1]
                    R[1,0,0,0] = XPC * R[0,0,0,1]
                    R[1,0,1,0] = XPC * R[0,0,1,1]
                    R[1,1,0,0] = XPC * R[0,1,0,1]
                    R[1,0,0,1] = XPC * R[0,0,0,2]
                    R[2,0,0,0] = 1 * R[0,0,0,1] + XPC * R[1,0,0,1]

                    primitives_buffer[i,j,k,l,0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,1,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,1] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,1,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,2] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,1,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,3] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,1,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,4] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,1,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,5] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,1,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,6] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,1,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,7] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,1,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,8] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,1,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,0]*norm_array[0,0,0]

                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]

    for out in range(0, 9):
        for i in range(0, number_primitive_1):
            for j in range(0, number_primitive_2):
                for k in range(0, number_primitive_3):
                    for l in range(0, number_primitive_4):
                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_1_1_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
            E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
            E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
            E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
            E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
            E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    XPA_right = P_right - Coord_3
                    XPB_right = P_right - Coord_4
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
                    
                    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
                    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
                    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]

                    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
                    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
                    R[0,0,1,0] = ZPC * R[0,0,0,1]
                    R[0,0,0,2] = (-2.0*alpha)**2 * boys_function(2,alpha*RPC*RPC)
                    R[0,0,1,1] = ZPC * R[0,0,0,2]
                    R[0,0,2,0] = 1 * R[0,0,0,1] + ZPC * R[0,0,1,1]
                    R[0,0,0,3] = (-2.0*alpha)**3 * boys_function(3,alpha*RPC*RPC)
                    R[0,0,1,2] = ZPC * R[0,0,0,3]
                    R[0,0,2,1] = 1 * R[0,0,0,2] + ZPC * R[0,0,1,2]
                    R[0,0,3,0] = 2 * R[0,0,1,1] + ZPC * R[0,0,2,1]
                    R[0,1,0,0] = YPC * R[0,0,0,1]
                    R[0,1,1,0] = YPC * R[0,0,1,1]
                    R[0,1,2,0] = YPC * R[0,0,2,1]
                    R[0,1,0,1] = YPC * R[0,0,0,2]
                    R[0,2,0,0] = 1 * R[0,0,0,1] + YPC * R[0,1,0,1]
                    R[0,1,1,1] = YPC * R[0,0,1,2]
                    R[0,2,1,0] = 1 * R[0,0,1,1] + YPC * R[0,1,1,1]
                    R[0,1,0,2] = YPC * R[0,0,0,3]
                    R[0,2,0,1] = 1 * R[0,0,0,2] + YPC * R[0,1,0,2]
                    R[0,3,0,0] = 2 * R[0,1,0,1] + YPC * R[0,2,0,1]
                    R[1,0,0,0] = XPC * R[0,0,0,1]
                    R[1,0,1,0] = XPC * R[0,0,1,1]
                    R[1,0,2,0] = XPC * R[0,0,2,1]
                    R[1,1,0,0] = XPC * R[0,1,0,1]
                    R[1,1,1,0] = XPC * R[0,1,1,1]
                    R[1,2,0,0] = XPC * R[0,2,0,1]
                    R[1,0,0,1] = XPC * R[0,0,0,2]
                    R[2,0,0,0] = 1 * R[0,0,0,1] + XPC * R[1,0,0,1]
                    R[1,0,1,1] = XPC * R[0,0,1,2]
                    R[2,0,1,0] = 1 * R[0,0,1,1] + XPC * R[1,0,1,1]
                    R[1,1,0,1] = XPC * R[0,1,0,2]
                    R[2,1,0,0] = 1 * R[0,1,0,1] + XPC * R[1,1,0,1]
                    R[1,0,0,2] = XPC * R[0,0,0,3]
                    R[2,0,0,1] = 1 * R[0,0,0,2] + XPC * R[1,0,0,2]
                    R[3,0,0,0] = 2 * R[1,0,0,1] + XPC * R[2,0,0,1]

                    primitives_buffer[i,j,k,l,0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,1,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,1] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,2,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,2] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,1,1,2)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,3] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,1,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,4] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,2,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,5] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,1,2)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,6] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,1,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,7] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,2,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,8] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,1,2)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,9] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,1,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,10] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,2,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,11] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,1,2)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,12] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,1,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,13] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,2,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,14] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,1,1,2)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,15] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,1,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,16] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,2,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,17] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,1,2)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,18] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,1,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,19] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,2,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,20] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,1,2)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,21] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,1,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,22] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,2,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,23] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,1,2)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,24] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,1,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,25] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,2,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,0]
                    primitives_buffer[i,j,k,l,26] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,1,1,2)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,0]

                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]

    for out in range(0, 27):
        for i in range(0, number_primitive_1):
            for j in range(0, number_primitive_2):
                for k in range(0, number_primitive_3):
                    for l in range(0, number_primitive_4):
                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)
def electron_electron_integral_1_1_1_1(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    number_primitive_3 = gauss_exp_3.shape[0]
    number_primitive_4 = gauss_exp_4.shape[0]
    pi = 3.141592653589793238462643383279
    pi52 = 2.0*pi**(5.0/2.0)
    XAB_left = Coord_1 - Coord_2
    XAB_right = Coord_3 - Coord_4
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = gauss_exp_1_left + gauss_exp_2_left
            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
            E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
            E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
            E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
            E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
            E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
            E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

            for k in range(0, number_primitive_3):
                gauss_exp_1_right = gauss_exp_3[k]
                for l in range(0, number_primitive_4):
                    gauss_exp_2_right = gauss_exp_4[l]
                    p_right = gauss_exp_1_right + gauss_exp_2_right
                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right
                    XPA_right = P_right - Coord_3
                    XPB_right = P_right - Coord_4
                    alpha = p_left*p_right/(p_left+p_right)
                    XPC, YPC, ZPC = P_left - P_right
                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
                    
                    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
                    E_right[0,1,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
                    E_right[0,1,0,:] = XPB_right * E_right[0,0,0,:]
                    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
                    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]
                    E_right[1,1,2,:] = (1.0/(2.0*p_right)) * E_right[1,0,1,:]
                    E_right[1,1,1,:] = (1.0/(2.0*p_right)) * E_right[1,0,0,:] + XPB_right * E_right[1,0,1,:]
                    E_right[1,1,0,:] = XPB_right * E_right[1,0,0,:] + 1.0 * E_right[1,0,1,:]

                    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
                    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
                    R[0,0,1,0] = ZPC * R[0,0,0,1]
                    R[0,0,0,2] = (-2.0*alpha)**2 * boys_function(2,alpha*RPC*RPC)
                    R[0,0,1,1] = ZPC * R[0,0,0,2]
                    R[0,0,2,0] = 1 * R[0,0,0,1] + ZPC * R[0,0,1,1]
                    R[0,0,0,3] = (-2.0*alpha)**3 * boys_function(3,alpha*RPC*RPC)
                    R[0,0,1,2] = ZPC * R[0,0,0,3]
                    R[0,0,2,1] = 1 * R[0,0,0,2] + ZPC * R[0,0,1,2]
                    R[0,0,3,0] = 2 * R[0,0,1,1] + ZPC * R[0,0,2,1]
                    R[0,0,0,4] = (-2.0*alpha)**4 * boys_function(4,alpha*RPC*RPC)
                    R[0,0,1,3] = ZPC * R[0,0,0,4]
                    R[0,0,2,2] = 1 * R[0,0,0,3] + ZPC * R[0,0,1,3]
                    R[0,0,3,1] = 2 * R[0,0,1,2] + ZPC * R[0,0,2,2]
                    R[0,0,4,0] = 3 * R[0,0,2,1] + ZPC * R[0,0,3,1]
                    R[0,1,0,0] = YPC * R[0,0,0,1]
                    R[0,1,1,0] = YPC * R[0,0,1,1]
                    R[0,1,2,0] = YPC * R[0,0,2,1]
                    R[0,1,3,0] = YPC * R[0,0,3,1]
                    R[0,1,0,1] = YPC * R[0,0,0,2]
                    R[0,2,0,0] = 1 * R[0,0,0,1] + YPC * R[0,1,0,1]
                    R[0,1,1,1] = YPC * R[0,0,1,2]
                    R[0,2,1,0] = 1 * R[0,0,1,1] + YPC * R[0,1,1,1]
                    R[0,1,2,1] = YPC * R[0,0,2,2]
                    R[0,2,2,0] = 1 * R[0,0,2,1] + YPC * R[0,1,2,1]
                    R[0,1,0,2] = YPC * R[0,0,0,3]
                    R[0,2,0,1] = 1 * R[0,0,0,2] + YPC * R[0,1,0,2]
                    R[0,3,0,0] = 2 * R[0,1,0,1] + YPC * R[0,2,0,1]
                    R[0,1,1,2] = YPC * R[0,0,1,3]
                    R[0,2,1,1] = 1 * R[0,0,1,2] + YPC * R[0,1,1,2]
                    R[0,3,1,0] = 2 * R[0,1,1,1] + YPC * R[0,2,1,1]
                    R[0,1,0,3] = YPC * R[0,0,0,4]
                    R[0,2,0,2] = 1 * R[0,0,0,3] + YPC * R[0,1,0,3]
                    R[0,3,0,1] = 2 * R[0,1,0,2] + YPC * R[0,2,0,2]
                    R[0,4,0,0] = 3 * R[0,2,0,1] + YPC * R[0,3,0,1]
                    R[1,0,0,0] = XPC * R[0,0,0,1]
                    R[1,0,1,0] = XPC * R[0,0,1,1]
                    R[1,0,2,0] = XPC * R[0,0,2,1]
                    R[1,0,3,0] = XPC * R[0,0,3,1]
                    R[1,1,0,0] = XPC * R[0,1,0,1]
                    R[1,1,1,0] = XPC * R[0,1,1,1]
                    R[1,1,2,0] = XPC * R[0,1,2,1]
                    R[1,2,0,0] = XPC * R[0,2,0,1]
                    R[1,2,1,0] = XPC * R[0,2,1,1]
                    R[1,3,0,0] = XPC * R[0,3,0,1]
                    R[1,0,0,1] = XPC * R[0,0,0,2]
                    R[2,0,0,0] = 1 * R[0,0,0,1] + XPC * R[1,0,0,1]
                    R[1,0,1,1] = XPC * R[0,0,1,2]
                    R[2,0,1,0] = 1 * R[0,0,1,1] + XPC * R[1,0,1,1]
                    R[1,0,2,1] = XPC * R[0,0,2,2]
                    R[2,0,2,0] = 1 * R[0,0,2,1] + XPC * R[1,0,2,1]
                    R[1,1,0,1] = XPC * R[0,1,0,2]
                    R[2,1,0,0] = 1 * R[0,1,0,1] + XPC * R[1,1,0,1]
                    R[1,1,1,1] = XPC * R[0,1,1,2]
                    R[2,1,1,0] = 1 * R[0,1,1,1] + XPC * R[1,1,1,1]
                    R[1,2,0,1] = XPC * R[0,2,0,2]
                    R[2,2,0,0] = 1 * R[0,2,0,1] + XPC * R[1,2,0,1]
                    R[1,0,0,2] = XPC * R[0,0,0,3]
                    R[2,0,0,1] = 1 * R[0,0,0,2] + XPC * R[1,0,0,2]
                    R[3,0,0,0] = 2 * R[1,0,0,1] + XPC * R[2,0,0,1]
                    R[1,0,1,2] = XPC * R[0,0,1,3]
                    R[2,0,1,1] = 1 * R[0,0,1,2] + XPC * R[1,0,1,2]
                    R[3,0,1,0] = 2 * R[1,0,1,1] + XPC * R[2,0,1,1]
                    R[1,1,0,2] = XPC * R[0,1,0,3]
                    R[2,1,0,1] = 1 * R[0,1,0,2] + XPC * R[1,1,0,2]
                    R[3,1,0,0] = 2 * R[1,1,0,1] + XPC * R[2,1,0,1]
                    R[1,0,0,3] = XPC * R[0,0,0,4]
                    R[2,0,0,2] = 1 * R[0,0,0,3] + XPC * R[1,0,0,3]
                    R[3,0,0,1] = 2 * R[1,0,0,2] + XPC * R[2,0,0,2]
                    R[4,0,0,0] = 3 * R[2,0,0,1] + XPC * R[3,0,0,1]

                    primitives_buffer[i,j,k,l,0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,3,1,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,1] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,2,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,2] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],3,1,1,2,1,2)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,3] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,2,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,4] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,3,1)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,5] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],3,1,1,1,2,2)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,6] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,2,1,2)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,7] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,1,2,2)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,8] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],3,1,1,1,1,3)*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,9] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,3,1,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,10] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,11] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,2,1,2)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,12] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,13] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,3,1)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,14] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,1,2,2)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,15] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,2,1,2)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,16] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,2,2)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,17] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,2,1,1,1,3)*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,18] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,3,1,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,19] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,20] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,2,1,2)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,21] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,22] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,3,1)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,23] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,1,2,2)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,24] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,2,1,2)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,25] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,2,2)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,26] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,1,2,1,1,3)*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,27] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,3,1,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,28] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,29] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,2,1,2)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,30] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,31] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,3,1)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,32] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,1,2,2)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,33] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,2,1,2)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,34] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,2,2)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,35] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,2,1,1,1,3)*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,36] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,3,1,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,37] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,2,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,38] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,3,1,2,1,2)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,39] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,2,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,40] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,3,1)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,41] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,3,1,1,2,2)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,42] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,2,1,2)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,43] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,1,2,2)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,44] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,3,1,1,1,3)*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,45] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,3,1,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,46] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,47] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,2,1,2)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,48] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,49] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,3,1)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,50] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,1,2,2)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,51] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,2,1,2)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,52] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,2,2)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,53] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,2,2,1,1,3)*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,54] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,3,1,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,55] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,56] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,2,1,2)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,57] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,58] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,3,1)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,59] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,1,2,2)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,60] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,2,1,2)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,61] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,2,2)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,62] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,1,2,1,1,3)*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,63] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,3,1,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,64] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,65] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,2,1,2)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,66] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,67] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,3,1)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,68] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,1,2,2)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,69] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,2,1,2)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,70] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,2,2)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,71] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,2,2,1,1,3)*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,72] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,3,1,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,73] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,2,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,74] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,1,3,2,1,2)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,75] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,2,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,76] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,3,1)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,77] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,1,3,1,2,2)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]*norm_array[0,0,1]
                    primitives_buffer[i,j,k,l,78] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,2,1,2)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[1,0,0]
                    primitives_buffer[i,j,k,l,79] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,1,2,2)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,1,0]
                    primitives_buffer[i,j,k,l,80] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,1,3,1,1,3)*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]*norm_array[0,0,1]

                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]

    for out in range(0, 81):
        for i in range(0, number_primitive_1):
            for j in range(0, number_primitive_2):
                for k in range(0, number_primitive_3):
                    for l in range(0, number_primitive_4):
                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]
    return output_buffer


