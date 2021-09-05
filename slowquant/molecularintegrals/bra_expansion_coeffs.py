import numpy as np
from slowquant.molecularintegrals.expansion_coefficients import *

def bra_expansion_coeffs_0_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    XAB_left = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = bra_array[0,i,j] = gauss_exp_1_left + gauss_exp_2_left
            q_left = bra_array[1,i,j] = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = bra_array[2:5,i,j] = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            p12_left = 1.0/(2.0*p_left)
            E_buff_1[i,j] = E_0_0_0(q_left, p12_left, XAB_left, XPA_left, XPB_left, E_buff_1[i,j])
    return E_buff_1, bra_array


def bra_expansion_coeffs_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    XAB_left = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = bra_array[0,i,j] = gauss_exp_1_left + gauss_exp_2_left
            q_left = bra_array[1,i,j] = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = bra_array[2:5,i,j] = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            p12_left = 1.0/(2.0*p_left)
            E_buff_1[i,j] = E_1_0_1(q_left, p12_left, XAB_left, XPA_left, XPB_left, E_buff_1[i,j])
    return E_buff_1, bra_array


def bra_expansion_coeffs_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):
    number_primitive_1 = gauss_exp_1.shape[0]
    number_primitive_2 = gauss_exp_2.shape[0]
    XAB_left = Coord_1 - Coord_2
    for i in range(0, number_primitive_1):
        gauss_exp_1_left = gauss_exp_1[i]
        for j in range(0, number_primitive_2):
            gauss_exp_2_left = gauss_exp_2[j]
            p_left = bra_array[0,i,j] = gauss_exp_1_left + gauss_exp_2_left
            q_left = bra_array[1,i,j] = gauss_exp_1_left * gauss_exp_2_left / p_left
            P_left = bra_array[2:5,i,j] = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left
            XPA_left = P_left - Coord_1
            XPB_left = P_left - Coord_2
            p12_left = 1.0/(2.0*p_left)
            E_buff_1[i,j] = E_1_1_2(q_left, p12_left, XAB_left, XPA_left, XPB_left, E_buff_1[i,j])
    return E_buff_1, bra_array


