import numpy as np
from numba import jit, float64
from slowquant.molecularintegrals.utility import Normalization, boys_function, ERI_expansion_coeff_sum, ERI_contraction


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_0_0_0_0(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)

    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)

    primitive[0] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,1,1,1,1)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_1_0_0_0(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
    E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
    E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]

    R[0,0,0,0] = (-2.0*alpha)**0 * boys_function(0,alpha*RPC*RPC)
    R[0,0,0,1] = (-2.0*alpha)**1 * boys_function(1,alpha*RPC*RPC)
    R[0,0,1,0] = ZPC * R[0,0,0,1]
    R[0,1,0,0] = YPC * R[0,0,0,1]
    R[1,0,0,0] = XPC * R[0,0,0,1]

    primitive[0] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,1,1,1)
    primitive[1] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,1,1,1)
    primitive[2] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,1,1,1)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_1_0_1_0(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_right[0,1,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[0,1,0,:] = XPB_right * E_right[0,0,0,:]
    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
    E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
    E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]

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

    primitive[0] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,2,1,1)
    primitive[1] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,1,1,2,1)
    primitive[2] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,1,1,1,2)
    primitive[3] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,2,1,1)
    primitive[4] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,1,1,2,1)
    primitive[5] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,1,1,1,2)
    primitive[6] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,2,1,1)
    primitive[7] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,2,1,2,1)
    primitive[8] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,2,1,1,2)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_1_1_0_0(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
    E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
    E_left[0,2,2,:] = (1.0/(2.0*p_left)) * E_left[0,1,1,:]
    E_left[0,2,1,:] = (1.0/(2.0*p_left)) * E_left[0,1,0,:] + XPB_left * E_left[0,1,1,:]
    E_left[0,2,0,:] = XPB_left * E_left[0,1,0,:] + 1.0 * E_left[0,1,1,:]
    E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
    E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
    E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]
    E_left[2,0,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[2,0,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPA_left * E_left[1,0,1,:]
    E_left[2,0,0,:] = XPA_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

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

    primitive[0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,1,1)
    primitive[1] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,1,1)
    primitive[2] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,1,1)
    primitive[3] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,1,1)
    primitive[4] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,1,1)
    primitive[5] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,1,1)
    primitive[6] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,1,1)
    primitive[7] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,1,1)
    primitive[8] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,1,1)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_1_1_1_0(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_right[0,1,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[0,1,0,:] = XPB_right * E_right[0,0,0,:]
    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
    E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
    E_left[0,2,2,:] = (1.0/(2.0*p_left)) * E_left[0,1,1,:]
    E_left[0,2,1,:] = (1.0/(2.0*p_left)) * E_left[0,1,0,:] + XPB_left * E_left[0,1,1,:]
    E_left[0,2,0,:] = XPB_left * E_left[0,1,0,:] + 1.0 * E_left[0,1,1,:]
    E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
    E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
    E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]
    E_left[2,0,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[2,0,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPA_left * E_left[1,0,1,:]
    E_left[2,0,0,:] = XPA_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

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

    primitive[0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,1,1)
    primitive[1] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,2,1)
    primitive[2] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,1,1,2)
    primitive[3] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,1,1)
    primitive[4] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,2,1)
    primitive[5] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,1,2)
    primitive[6] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,1,1)
    primitive[7] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,2,1)
    primitive[8] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,1,2)
    primitive[9] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,1,1)
    primitive[10] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,2,1)
    primitive[11] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,1,2)
    primitive[12] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,1,1)
    primitive[13] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,2,1)
    primitive[14] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,1,1,2)
    primitive[15] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,1,1)
    primitive[16] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,2,1)
    primitive[17] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,1,2)
    primitive[18] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,1,1)
    primitive[19] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,2,1)
    primitive[20] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,1,2)
    primitive[21] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,1,1)
    primitive[22] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,2,1)
    primitive[23] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,1,2)
    primitive[24] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,1,1)
    primitive[25] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,2,1)
    primitive[26] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,1,1,2)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64, float64, float64, float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_electron_electron_1_1_1_1(Coord_1_left, Coord_2_left, Coord_1_right, Coord_2_right, gauss_exp_1_left, gauss_exp_2_left, gauss_exp_1_right, gauss_exp_2_right, E_left, E_right, R, primitive):
    pi = 3.141592653589793238462643383279
    p_left = gauss_exp_1_left + gauss_exp_2_left
    q_left = gauss_exp_1_left * gauss_exp_2_left / p_left
    P_left = (gauss_exp_1_left*Coord_1_left + gauss_exp_2_left*Coord_2_left) / p_left
    XAB_left = Coord_1_left - Coord_2_left
    XPA_left = P_left - Coord_1_left
    XPB_left = P_left - Coord_2_left
    p_right = gauss_exp_1_right + gauss_exp_2_right
    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right
    P_right = (gauss_exp_1_right*Coord_1_right + gauss_exp_2_right*Coord_2_right) / p_right
    XAB_right = Coord_1_right - Coord_2_right
    XPA_right = P_right - Coord_1_right
    XPB_right = P_right - Coord_2_right
    alpha = p_left*p_right/(p_left+p_right)
    XPC, YPC, ZPC = P_left - P_right
    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5
    primitive[:] = 0.0
    
    E_right[0,0,0,:] = np.exp(-q_right*XAB_right*XAB_right)
    E_right[0,1,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[0,1,0,:] = XPB_right * E_right[0,0,0,:]
    E_right[0,2,2,:] = (1.0/(2.0*p_right)) * E_right[0,1,1,:]
    E_right[0,2,1,:] = (1.0/(2.0*p_right)) * E_right[0,1,0,:] + XPB_right * E_right[0,1,1,:]
    E_right[0,2,0,:] = XPB_right * E_right[0,1,0,:] + 1.0 * E_right[0,1,1,:]
    E_right[1,0,1,:] = (1.0/(2.0*p_right)) * E_right[0,0,0,:]
    E_right[1,0,0,:] = XPA_right * E_right[0,0,0,:]
    E_right[1,1,2,:] = (1.0/(2.0*p_right)) * E_right[1,0,1,:]
    E_right[1,1,1,:] = (1.0/(2.0*p_right)) * E_right[1,0,0,:] + XPB_right * E_right[1,0,1,:]
    E_right[1,1,0,:] = XPB_right * E_right[1,0,0,:] + 1.0 * E_right[1,0,1,:]
    E_right[2,0,2,:] = (1.0/(2.0*p_right)) * E_right[1,0,1,:]
    E_right[2,0,1,:] = (1.0/(2.0*p_right)) * E_right[1,0,0,:] + XPA_right * E_right[1,0,1,:]
    E_right[2,0,0,:] = XPA_right * E_right[1,0,0,:] + 1.0 * E_right[1,0,1,:]
    E_left[0,0,0,:] = np.exp(-q_left*XAB_left*XAB_left)
    E_left[0,1,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[0,1,0,:] = XPB_left * E_left[0,0,0,:]
    E_left[0,2,2,:] = (1.0/(2.0*p_left)) * E_left[0,1,1,:]
    E_left[0,2,1,:] = (1.0/(2.0*p_left)) * E_left[0,1,0,:] + XPB_left * E_left[0,1,1,:]
    E_left[0,2,0,:] = XPB_left * E_left[0,1,0,:] + 1.0 * E_left[0,1,1,:]
    E_left[1,0,1,:] = (1.0/(2.0*p_left)) * E_left[0,0,0,:]
    E_left[1,0,0,:] = XPA_left * E_left[0,0,0,:]
    E_left[1,1,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[1,1,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPB_left * E_left[1,0,1,:]
    E_left[1,1,0,:] = XPB_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]
    E_left[2,0,2,:] = (1.0/(2.0*p_left)) * E_left[1,0,1,:]
    E_left[2,0,1,:] = (1.0/(2.0*p_left)) * E_left[1,0,0,:] + XPA_left * E_left[1,0,1,:]
    E_left[2,0,0,:] = XPA_left * E_left[1,0,0,:] + 1.0 * E_left[1,0,1,:]

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

    primitive[0] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,3,1,1)
    primitive[1] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,2,1)
    primitive[2] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],3,1,1,2,1,2)
    primitive[3] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,2,2,1)
    primitive[4] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],3,1,1,1,3,1)
    primitive[5] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],3,1,1,1,2,2)
    primitive[6] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,2,1,2)
    primitive[7] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],3,1,1,1,2,2)
    primitive[8] = ERI_expansion_coeff_sum(E_left[1,1,:,0],E_left[0,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],3,1,1,1,1,3)
    primitive[9] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,3,1,1)
    primitive[10] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)
    primitive[11] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,2,1,2)
    primitive[12] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)
    primitive[13] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,3,1)
    primitive[14] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,1,2,2)
    primitive[15] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,2,1,2)
    primitive[16] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,2,2)
    primitive[17] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,2,1,1,1,3)
    primitive[18] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,3,1,1)
    primitive[19] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)
    primitive[20] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,2,1,2)
    primitive[21] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)
    primitive[22] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,3,1)
    primitive[23] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,1,2,2)
    primitive[24] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,2,1,2)
    primitive[25] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,2,2)
    primitive[26] = ERI_expansion_coeff_sum(E_left[1,0,:,0],E_left[0,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,1,2,1,1,3)
    primitive[27] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,3,1,1)
    primitive[28] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)
    primitive[29] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,2,1,2)
    primitive[30] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,2,2,1)
    primitive[31] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,2,1,1,3,1)
    primitive[32] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,2,1,1,2,2)
    primitive[33] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,2,1,2)
    primitive[34] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,2,1,1,2,2)
    primitive[35] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[1,0,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,2,1,1,1,3)
    primitive[36] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,3,1,1)
    primitive[37] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,2,1)
    primitive[38] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,3,1,2,1,2)
    primitive[39] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,2,2,1)
    primitive[40] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,3,1,1,3,1)
    primitive[41] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,3,1,1,2,2)
    primitive[42] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,2,1,2)
    primitive[43] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,3,1,1,2,2)
    primitive[44] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,1,:,1],E_left[0,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,3,1,1,1,3)
    primitive[45] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,3,1,1)
    primitive[46] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)
    primitive[47] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,2,1,2)
    primitive[48] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)
    primitive[49] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,3,1)
    primitive[50] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,1,2,2)
    primitive[51] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,2,1,2)
    primitive[52] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,2,2)
    primitive[53] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[1,0,:,1],E_left[0,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,2,2,1,1,3)
    primitive[54] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,3,1,1)
    primitive[55] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)
    primitive[56] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,2,1,2)
    primitive[57] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,2,2,1)
    primitive[58] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],2,1,2,1,3,1)
    primitive[59] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],2,1,2,1,2,2)
    primitive[60] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,2,1,2)
    primitive[61] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],2,1,2,1,2,2)
    primitive[62] = ERI_expansion_coeff_sum(E_left[0,1,:,0],E_left[0,0,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],2,1,2,1,1,3)
    primitive[63] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,3,1,1)
    primitive[64] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)
    primitive[65] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,2,1,2)
    primitive[66] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,2,2,1)
    primitive[67] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,2,2,1,3,1)
    primitive[68] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,2,2,1,2,2)
    primitive[69] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,2,1,2)
    primitive[70] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,2,2,1,2,2)
    primitive[71] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,1,:,1],E_left[1,0,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,2,2,1,1,3)
    primitive[72] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,1,:,0],E_right[0,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,3,1,1)
    primitive[73] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,2,1)
    primitive[74] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[1,0,:,0],E_right[0,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,1,3,2,1,2)
    primitive[75] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,1,:,0],E_right[1,0,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,2,2,1)
    primitive[76] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,1,:,1],E_right[0,0,:,2],R[:,:,:,0],1,1,3,1,3,1)
    primitive[77] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[1,0,:,1],E_right[0,1,:,2],R[:,:,:,0],1,1,3,1,2,2)
    primitive[78] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,1,:,0],E_right[0,0,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,2,1,2)
    primitive[79] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,1,:,1],E_right[1,0,:,2],R[:,:,:,0],1,1,3,1,2,2)
    primitive[80] = ERI_expansion_coeff_sum(E_left[0,0,:,0],E_left[0,0,:,1],E_left[1,1,:,2],E_right[0,0,:,0],E_right[0,0,:,1],E_right[1,1,:,2],R[:,:,:,0],1,1,3,1,1,3)

    return 2.0*pi**(5.0/2.0)/(p_left*p_right*(p_left+p_right)**0.5)*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_0_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:1]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_0_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 1):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_1_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:3]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_1_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 3):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_1_0_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:9]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_1_0_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 9):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_1_1_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:9]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_1_1_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 9):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_1_1_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:27]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_1_1_1_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 27):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def electron_electron_integral_1_1_1_1(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2, output_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    number_primitive_3 = len(gauss_exp_3)
    number_primitive_4 = len(gauss_exp_4)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    Contraction_3_buffer = Contraction_3_buffer[:number_primitive_3]
    Contraction_4_buffer = Contraction_4_buffer[:number_primitive_4]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:number_primitive_3,:number_primitive_4,:81]
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            for k in range(0, number_primitive_3):
                for l in range(0, number_primitive_4):
                    primitives_buffer[i,j,k,l,:] = primitive_electron_electron_1_1_1_1(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1[i], gauss_exp_2[j], gauss_exp_3[k], gauss_exp_4[l], E_1_buffer, E_2_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, number_primitive_1):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, number_primitive_2):
        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    for i in range(0, number_primitive_3):
        Contraction_3_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_3[i]) * Contra_coeffs_3[i]
    for i in range(0, number_primitive_4):
        Contraction_4_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_4[i]) * Contra_coeffs_4[i]
    for i in range(0, 81):
        output_buffer[i] = ERI_contraction(Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, primitives_buffer[:,:,:,:,i])
    return output_buffer


