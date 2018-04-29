import numpy as np
from numba import jit, float64
from slowquant.molecularintegrals.utility import Normalization


@jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)
def primitive_overlap_0_0(Coord_1, Coord_2, guass_exp_1, gauss_exp_2):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    E_0_0_0 = np.exp(-q*XAB*XAB)

    return np.array([(pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_0_0_0[2]])


@jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)
def primitive_overlap_1_0(Coord_1, Coord_2, guass_exp_1, gauss_exp_2):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    E_0_0_0 = np.exp(-q*XAB*XAB)
    E_1_0_0 = XPA * E_0_0_0

    return np.array([(pi/p)**(3/2) * E_1_0_0[0] * E_0_0_0[1] * E_0_0_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_0_0[1] * E_0_0_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_1_0_0[2]])


@jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)
def primitive_overlap_1_1(Coord_1, Coord_2, guass_exp_1, gauss_exp_2):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    E_0_0_0 = np.exp(-q*XAB*XAB)
    E_0_1_0 = XPB * E_0_0_0
    E_1_0_0 = XPA * E_0_0_0
    E_1_0_1 = (1.0/(2.0*p)) * E_0_0_0
    E_1_1_0 = XPB * E_1_0_0 + 1.0 * E_1_0_1

    return np.array([(pi/p)**(3/2) * E_1_1_0[0] * E_0_0_0[1] * E_0_0_0[2],
                     (pi/p)**(3/2) * E_1_0_0[0] * E_0_1_0[1] * E_0_0_0[2],
                     (pi/p)**(3/2) * E_1_0_0[0] * E_0_0_0[1] * E_0_1_0[2],
                     (pi/p)**(3/2) * E_0_1_0[0] * E_1_0_0[1] * E_0_1_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_1_0[1] * E_0_0_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_0_0[1] * E_0_1_0[2],
                     (pi/p)**(3/2) * E_0_1_0[0] * E_0_0_0[1] * E_1_0_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_1_0[1] * E_1_0_0[2],
                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_1_1_0[2]])


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def overlap_integral_0_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:1]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_overlap_0_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def overlap_integral_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:3]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_overlap_1_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])
    output_buffer[1] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])
    output_buffer[2] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:]), nopython=True, cache=True)
def overlap_integral_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:9]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_overlap_1_1(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])
    output_buffer[1] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])
    output_buffer[2] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,3])
    output_buffer[3] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,4])
    output_buffer[4] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,5])
    output_buffer[5] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,6])
    output_buffer[6] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,7])
    output_buffer[7] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,8])
    output_buffer[8] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


