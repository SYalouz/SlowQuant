import numpy as np


def write_bra_expansion_coeffs(max_angular):
    S_file = open("slowquant/molecularintegrals/bra_expansion_coeffs.py", "w+")
    S_file.write("import numpy as np\n")
    #S_file.write("from numba import jit, float64\n")
    #S_file.write("from numba.types import Tuple\n")
    S_file.write("from slowquant.molecularintegrals.expansion_coefficients import *\n")
    S_file.write("\n")
    for lb in range(max_angular+1):
        for la in range(lb, max_angular+1):
            #S_file.write("@jit(Tuple((float64[:,:,:,:,:,:],float64[:,:,:]))(float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:,:], float64[:,:,:]), nopython=True, cache=True)\n")
            S_file.write("def bra_expansion_coeffs_"+str(la)+"_"+str(lb)+"(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, E_buff_1, bra_array):\n")
            S_file.write("    number_primitive_1 = gauss_exp_1.shape[0]\n")
            S_file.write("    number_primitive_2 = gauss_exp_2.shape[0]\n")
            S_file.write("    XAB_left = Coord_1 - Coord_2\n")
            S_file.write("    for i in range(0, number_primitive_1):\n")
            S_file.write("        gauss_exp_1_left = gauss_exp_1[i]\n")
            S_file.write("        for j in range(0, number_primitive_2):\n")
            S_file.write("            gauss_exp_2_left = gauss_exp_2[j]\n")
            S_file.write("            p_left = bra_array[0,i,j] = gauss_exp_1_left + gauss_exp_2_left\n")
            S_file.write("            q_left = bra_array[1,i,j] = gauss_exp_1_left * gauss_exp_2_left / p_left\n")
            S_file.write("            P_left = bra_array[2:5,i,j] = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left\n")
            S_file.write("            XPA_left = P_left - Coord_1\n")
            S_file.write("            XPB_left = P_left - Coord_2\n")
            S_file.write("            p12_left = 1.0/(2.0*p_left)\n")
            S_file.write("            E_buff_1[i,j] = E_"+str(la)+"_"+str(lb)+"_"+str(la+lb)+"(q_left, p12_left, XAB_left, XPA_left, XPB_left, E_buff_1[i,j])\n")
            S_file.write("    return E_buff_1, bra_array\n")
            S_file.write("\n\n")