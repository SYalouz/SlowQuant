import numpy as np


def write_electron_electron(max_angular):
    S_file = open("../../slowquant/molecularintegrals/electron_electron_MD4.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.utility import ERI_expansion_coeff_sum, Contraction_two_electron\n")
    S_file.write("from slowquant.molecularintegrals.expansion_coefficients import *\n")
    S_file.write("from slowquant.molecularintegrals.hermite_integral import *\n")
    S_file.write("\n\n")
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                for lc in range(max_angular+1):
                    for ld in range(max_angular+1):
                        if lc >= ld and la*(la+1)//2+lb >= lc*(lc+1)//2+ld:
                            S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:,:,:]), nopython=True, cache=True)\n")
                            S_file.write("def electron_electron_integral_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"_MD4(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, E_buffer, R_buffer, output_buffer, Norm_array):\n")
                            S_file.write("    number_primitive_1 = gauss_exp_1.shape[0]\n")
                            S_file.write("    number_primitive_2 = gauss_exp_2.shape[0]\n")
                            S_file.write("    number_primitive_3 = gauss_exp_3.shape[0]\n")
                            S_file.write("    number_primitive_4 = gauss_exp_4.shape[0]\n")
                            S_file.write("    pi = 3.141592653589793238462643383279\n")
                            S_file.write("    pi52 = 2.0*pi**(5.0/2.0)\n")
                            S_file.write("    XAB_left = Coord_1 - Coord_2\n")
                            S_file.write("    XAB_right = Coord_3 - Coord_4\n")
                            S_file.write("    for i in range(0, number_primitive_1):\n")
                            S_file.write("        gauss_exp_1_left = gauss_exp_1[i]\n")
                            S_file.write("        for j in range(0, number_primitive_2):\n")
                            S_file.write("            gauss_exp_2_left = gauss_exp_2[j]\n")
                            S_file.write("            p_left = gauss_exp_1_left + gauss_exp_2_left\n")
                            S_file.write("            q_left = gauss_exp_1_left * gauss_exp_2_left / p_left\n")
                            S_file.write("            P_left = (gauss_exp_1_left*Coord_1 + gauss_exp_2_left*Coord_2) / p_left\n")
                            S_file.write("            XPA_left = P_left - Coord_1\n")
                            S_file.write("            XPB_left = P_left - Coord_2\n")
                            S_file.write("            p12_left = 1.0/(2.0*p_left)\n")
                            S_file.write("            E1 = E_"+str(la)+"_"+str(lb)+"_"+str(la+lb)+"(q_left, p12_left, XAB_left, XPA_left, XPB_left, E_buffer)\n")
                            S_file.write("            for k in range(0, number_primitive_3):\n")
                            S_file.write("                gauss_exp_1_right = gauss_exp_3[k]\n")
                            S_file.write("                for l in range(0, number_primitive_4):\n")
                            S_file.write("                    gauss_exp_2_right = gauss_exp_4[l]\n")
                            S_file.write("                    p_right = gauss_exp_1_right + gauss_exp_2_right\n")
                            S_file.write("                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right\n")
                            S_file.write("                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right\n")
                            S_file.write("                    XPA_right = P_right - Coord_3\n")
                            S_file.write("                    XPB_right = P_right - Coord_4\n")
                            S_file.write("                    alpha = p_left*p_right/(p_left+p_right)\n")
                            S_file.write("                    XPC, YPC, ZPC = P_left - P_right\n")
                            S_file.write("                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5\n")
                            S_file.write("                    p12_right = 1.0/(2.0*p_right)\n")
                            S_file.write("                    E2 = E_"+str(lc)+"_"+str(ld)+"_"+str(lc+ld)+"(q_right, p12_right, XAB_right, XPA_right, XPB_right, E_buffer)\n")
                            S_file.write("                    R_array = R_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"(alpha, XPC, YPC, ZPC, RPC, R_buffer)\n")
                            S_file.write("                    counter = 0\n")
                            S_file.write("                    for x1 in range("+str(la)+", -1, -1):\n")
                            S_file.write("                        for y1 in range("+str(la)+"-x1, -1, -1):\n")
                            S_file.write("                            for z1 in range("+str(la)+"-x1-y1, "+str(la-1)+"-x1-y1, -1):\n")
                            S_file.write("                                temp1 = Norm_array[x1, y1, z1]\n")
                            S_file.write("                                for x2 in range("+str(lb)+", -1, -1):\n")
                            S_file.write("                                    for y2 in range("+str(lb)+"-x2, -1, -1):\n")
                            S_file.write("                                        for z2 in range("+str(lb)+"-x2-y2, "+str(lb-1)+"-x2-y2, -1):\n")
                            S_file.write("                                            temp2 = temp1*Norm_array[x2, y2, z2]\n")
                            S_file.write("                                            for x3 in range("+str(lc)+", -1, -1):\n")
                            S_file.write("                                                for y3 in range("+str(lc)+"-x3, -1, -1):\n")
                            S_file.write("                                                    for z3 in range("+str(lc)+"-x3-y3, "+str(lc-1)+"-x3-y3, -1):\n")
                            S_file.write("                                                        temp3 = temp2*Norm_array[x3, y3, z3]\n")
                            S_file.write("                                                        for x4 in range("+str(ld)+", -1, -1):\n")
                            S_file.write("                                                            for y4 in range("+str(ld)+"-x4, -1, -1):\n")
                            S_file.write("                                                                for z4 in range("+str(ld)+"-x4-y3, "+str(ld-1)+"-x4-y4, -1):\n")
                            S_file.write("                                                                    primitives_buffer[i,j,k,l,counter] = ERI_expansion_coeff_sum(E1[x1,x2,:,0],E1[y1,y2,:,1],E1[z1,z2,:,2],E2[x3,x4,:,0],E2[y3,y4,:,1],E2[z3,z4,:,2],R_array,x1+x2+1,y1+y2+1,z1+z2+1,x3+x4+1,y3+y4+1,z3+z4+1)*temp3*Norm_array[x4, y4, z4]\n")
                            S_file.write("                                                                    counter += 1\n")
                            S_file.write("                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]\n")
                            S_file.write("    for i in range(0, "+str((la+1)*((la+1)+1)//2*(lb+1)*((lb+1)+1)//2*(lc+1)*((lc+1)+1)//2*(ld+1)*((ld+1)+1)//2)+"):\n")
                            S_file.write("        output_buffer[i] = Contraction_two_electron(primitives_buffer[:,:,:,:,i], Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4)\n")
                            S_file.write("    return output_buffer\n")
                            S_file.write("\n\n")
                            
write_electron_electron(1)