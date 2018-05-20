import numpy as np
from numba import jit, float64
from slowquant.molecularintegrals.expansion_coefficients import *
from slowquant.molecularintegrals.utility import Contraction_one_electron

@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:]), nopython=True, cache=True)
def overlap_integral_0_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Norm_array, E_buffer):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    pi = 3.141592653589793238462643383279
    XAB = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            p = gauss_exp_1[i] + gauss_exp_2[j]
            q = gauss_exp_1[i] * gauss_exp_2[j] / p
            P = (gauss_exp_1[i]*Coord_1 + gauss_exp_2[j]*Coord_2) / p
            XPA = P - Coord_1
            XPB = P - Coord_2
            p12 = 1.0/(2.0*p)
            E_buffer = E_0_0_0(q, p12, XAB, XPA, XPB, E_buffer)
            counter = 0
            for x1 in range(0, -1, -1):
                for y1 in range(0-x1, -1, -1):
                    for z1 in range(0-x1-y1, -1-x1-y1, -1):
                        temp1 = Norm_array[x1, y1, z1]
                        for x2 in range(0, -1, -1):
                            for y2 in range(0-x2, -1, -1):
                                for z2 in range(0-x2-y2, -1-x2-y2, -1):
                                   primitives_buffer[i,j,counter] = E_buffer[x1,x2,0,0]*E_buffer[y1,y2,0,1]*E_buffer[z1,z2,0,2]*temp1*Norm_array[x2,y2,z2]
                                   counter += 1
            primitives_buffer[i,j,:] = (pi/p)**(3.0/2.0)*primitives_buffer[i,j,:]

    for i in range(0, 1):
        output_buffer[i] = Contraction_one_electron(primitives_buffer[:,:,i], Contra_coeffs_1, Contra_coeffs_2)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:]), nopython=True, cache=True)
def overlap_integral_1_0_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Norm_array, E_buffer):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    pi = 3.141592653589793238462643383279
    XAB = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            p = gauss_exp_1[i] + gauss_exp_2[j]
            q = gauss_exp_1[i] * gauss_exp_2[j] / p
            P = (gauss_exp_1[i]*Coord_1 + gauss_exp_2[j]*Coord_2) / p
            XPA = P - Coord_1
            XPB = P - Coord_2
            p12 = 1.0/(2.0*p)
            E_buffer = E_1_0_0(q, p12, XAB, XPA, XPB, E_buffer)
            counter = 0
            for x1 in range(1, -1, -1):
                for y1 in range(1-x1, -1, -1):
                    for z1 in range(1-x1-y1, 0-x1-y1, -1):
                        temp1 = Norm_array[x1, y1, z1]
                        for x2 in range(0, -1, -1):
                            for y2 in range(0-x2, -1, -1):
                                for z2 in range(0-x2-y2, -1-x2-y2, -1):
                                   primitives_buffer[i,j,counter] = E_buffer[x1,x2,0,0]*E_buffer[y1,y2,0,1]*E_buffer[z1,z2,0,2]*temp1*Norm_array[x2,y2,z2]
                                   counter += 1
            primitives_buffer[i,j,:] = (pi/p)**(3.0/2.0)*primitives_buffer[i,j,:]

    for i in range(0, 3):
        output_buffer[i] = Contraction_one_electron(primitives_buffer[:,:,i], Contra_coeffs_1, Contra_coeffs_2)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:]), nopython=True, cache=True)
def overlap_integral_1_1_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Norm_array, E_buffer):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    pi = 3.141592653589793238462643383279
    XAB = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        for j in range(0, number_primitive_2):
            p = gauss_exp_1[i] + gauss_exp_2[j]
            q = gauss_exp_1[i] * gauss_exp_2[j] / p
            P = (gauss_exp_1[i]*Coord_1 + gauss_exp_2[j]*Coord_2) / p
            XPA = P - Coord_1
            XPB = P - Coord_2
            p12 = 1.0/(2.0*p)
            E_buffer = E_1_1_0(q, p12, XAB, XPA, XPB, E_buffer)
            counter = 0
            for x1 in range(1, -1, -1):
                for y1 in range(1-x1, -1, -1):
                    for z1 in range(1-x1-y1, 0-x1-y1, -1):
                        temp1 = Norm_array[x1, y1, z1]
                        for x2 in range(1, -1, -1):
                            for y2 in range(1-x2, -1, -1):
                                for z2 in range(1-x2-y2, 0-x2-y2, -1):
                                   primitives_buffer[i,j,counter] = E_buffer[x1,x2,0,0]*E_buffer[y1,y2,0,1]*E_buffer[z1,z2,0,2]*temp1*Norm_array[x2,y2,z2]
                                   counter += 1
            primitives_buffer[i,j,:] = (pi/p)**(3.0/2.0)*primitives_buffer[i,j,:]

    for i in range(0, 9):
        output_buffer[i] = Contraction_one_electron(primitives_buffer[:,:,i], Contra_coeffs_1, Contra_coeffs_2)
    return output_buffer


