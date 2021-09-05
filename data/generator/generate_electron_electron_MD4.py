import numpy as np
from numba import jit, float64

@jit(float64(float64), nopython=True, cache=False)
def factorial2(n):
    n_range = int(n)
    out = 1.0
    if n > 0:
        for i in range(0, int(n_range+1)//2):
            out = out*(n-2*i)
    return out
    

@jit(float64(float64, float64, float64), nopython=True, cache=False)
def PsuedoNorm2(l,m,n):
    pi = 3.141592653589793238462643383279
    # Normalize primitive functions
    part1 = (2.0/pi)**(3.0/4.0)*2.0**(l+m+n)
    part3 = (factorial2(int(2*l-1))*factorial2(int(2*m-1))*factorial2(int(2*n-1)))**0.5
    N = part1/part3
    return N


def write_electron_electron(max_angular):
    S_file = open("slowquant/molecularintegrals/electron_electron_MD4.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numpy import exp\n")
<<<<<<< HEAD
    S_file.write("from numba import jit, float64\n")
=======
    #S_file.write("from numba import jit, float64\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
    S_file.write("from slowquant.molecularintegrals.utility import ERI_expansion_coeff_sum, Contraction_two_electron, ERI_expansion_coeff_sum_X_X_S_S\n")
    S_file.write("from slowquant.molecularintegrals.expansion_coefficients import *\n")
    S_file.write("from slowquant.molecularintegrals.hermite_integral import *\n")
    S_file.write("from slowquant.molecularintegrals.bra_expansion_coeffs import *\n")
    S_file.write("\n\n")
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                for lc in range(max_angular+1):
                    for ld in range(max_angular+1):
                        if lc >= ld and la*(la+1)//2+lb >= lc*(lc+1)//2+ld:
                            combinations = (la+1)*((la+1)+1)//2*(lb+1)*((lb+1)+1)//2*(lc+1)*((lc+1)+1)//2*(ld+1)*((ld+1)+1)//2
<<<<<<< HEAD
                            S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:,:,:,:,:,:], float64[:,:,:,:,:,:], float64[:,:,:,:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:]), nopython=True, cache=True)\n")
=======
                            #S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:,:,:,:,:,:], float64[:,:,:,:,:,:], float64[:,:,:,:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:]), nopython=True, cache=True)\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
                            S_file.write("def electron_electron_integral_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"_MD4(Coord_3, Coord_4, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, E_buff_1, E_buff_2, R_buffer, output_buffer, Norm_array, bra_array, ket_array):\n")
                            S_file.write("    number_primitive_1 = Contra_coeffs_1.shape[0]\n")
                            S_file.write("    number_primitive_2 = Contra_coeffs_2.shape[0]\n")
                            S_file.write("    number_primitive_3 = Contra_coeffs_3.shape[0]\n")
                            S_file.write("    number_primitive_4 = Contra_coeffs_4.shape[0]\n")
                            #S_file.write("    pi = 3.141592653589793238462643383279\n")
                            #S_file.write("    pi52 = 2.0*pi**(5.0/2.0)\n")
                            S_file.write("    XAB_right = Coord_3 - Coord_4\n")
                            S_file.write("    for k in range(0, number_primitive_3):\n")
                            S_file.write("        gauss_exp_1_right = gauss_exp_3[k]\n")
                            S_file.write("        for l in range(0, number_primitive_4):\n")
                            S_file.write("            gauss_exp_2_right = gauss_exp_4[l]\n")
                            S_file.write("            p_right = ket_array[0,k,l] = gauss_exp_1_right + gauss_exp_2_right\n")
                            S_file.write("            q_right = ket_array[1,k,l] = gauss_exp_1_right * gauss_exp_2_right / p_right\n")
                            S_file.write("            P_right = ket_array[2:5,k,l] = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right\n")
                            if lc == 0 and ld == 0:
                                S_file.write("            E_buff_2[k,l,0,0,0,:] = exp(-q_right*XAB_right*XAB_right)\n")
                            else:
                                S_file.write("            XPA_right = P_right - Coord_3\n")
                                S_file.write("            XPB_right = P_right - Coord_4\n")
                                S_file.write("            p12_right = 1.0/(2.0*p_right)\n")
                                S_file.write("            E_buff_2[k,l] = E_"+str(lc)+"_"+str(ld)+"_"+str(lc+ld)+"(q_right, p12_right, XAB_right, XPA_right, XPB_right, E_buff_2[k,l])\n")
                            S_file.write("    for i in range(0, number_primitive_1):\n")
                            S_file.write("        for j in range(0, number_primitive_2):\n")
                            S_file.write("            p_left = bra_array[0,i,j]\n")
                            S_file.write("            q_left = bra_array[1,i,j]\n")
                            S_file.write("            P_left = bra_array[2:5,i,j]\n")
                            S_file.write("            for k in range(0, number_primitive_3):\n")
                            S_file.write("                for l in range(0, number_primitive_4):\n")
                            S_file.write("                    p_right = ket_array[0,k,l]\n")
                            S_file.write("                    q_right = ket_array[1,k,l]\n")
                            S_file.write("                    P_right = ket_array[2:5,k,l]\n")
                            S_file.write("                    alpha = p_left*p_right/(p_left+p_right)\n")
                            S_file.write("                    XPC, YPC, ZPC = P_left - P_right\n")
                            S_file.write("                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5\n")
                            S_file.write("                    R_array = R_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"(alpha, XPC, YPC, ZPC, RPC, R_buffer)\n")
                            S_file.write("                    counter = 0\n")
                            if lc == 0 and ld == 0 and la != 0:
                                S_file.write("                    primitives_buffer[i,j,k,l,:"+str(combinations)+"] = 0.0\n")
                            indentation = "                    "
                            if la != 0:
                                x1, y1, z1 = "x1", "y1", "z1"
                                # These loops goes up to angular moment. If angular moment is zero, the loop
                                #  will only slow down the compiler and the code.
                                S_file.write(indentation+"for x1 in range("+str(la)+", -1, -1):\n")
                                S_file.write(indentation+"    for y1 in range("+str(la)+"-x1, -1, -1):\n")
                                S_file.write(indentation+"        for z1 in range("+str(la)+"-x1-y1, "+str(la-1)+"-x1-y1, -1):\n")
                                indentation = indentation + "            "
                                if la > 1:
                                    # For angular moment 0 and 1, the second part of the normalization is the 
<<<<<<< HEAD
                                    #  same for all of the pritimives. To save some flops these can be 
                                    #  applied after the contraction, wheras higher angular momemntum have to be
=======
                                    #  same for all of the primitives. To save some flops these can be 
                                    #  applied after the contraction, whereas higher angular momentum have to be
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
                                    #  applied straight away.
                                    S_file.write(indentation+"temp1 = Norm_array[x1, y1, z1]\n")
                            else:
                                x1, y1, z1 = "0", "0", "0"
                            if lb != 0:
                                x2, y2, z2 = "x2", "y2", "z2"
                                S_file.write(indentation+"for x2 in range("+str(lb)+", -1, -1):\n")
                                S_file.write(indentation+"    for y2 in range("+str(lb)+"-x2, -1, -1):\n")
                                S_file.write(indentation+"        for z2 in range("+str(lb)+"-x2-y2, "+str(lb-1)+"-x2-y2, -1):\n")
                                indentation = indentation + "            "
                                if lb > 1 and la > 1:
                                    S_file.write(indentation+"temp2 = temp1*Norm_array[x2, y2, z2]\n")
                                elif lb > 1:
                                    S_file.write(indentation+"temp2 = Norm_array[x2, y2, z2]\n")
                            else:
                                x2, y2, z2 = "0", "0", "0"
                            if lc != 0:
                                x3, y3, z3 = "x3", "y3", "z3"
                                S_file.write(indentation+"for x3 in range("+str(lc)+", -1, -1):\n")
                                S_file.write(indentation+"    for y3 in range("+str(lc)+"-x3, -1, -1):\n")
                                S_file.write(indentation+"        for z3 in range("+str(lc)+"-x3-y3, "+str(lc-1)+"-x3-y3, -1):\n")
                                indentation = indentation + "            "
                                if lc > 1 and lb > 1:
                                    S_file.write(indentation+"temp3 = temp2*Norm_array[x3, y3, z3]\n")
                                elif lc > 1 and la > 1:
                                    S_file.write(indentation+"temp3 = temp1*Norm_array[x3, y3, z3]\n")
                                elif lc > 1:
                                    S_file.write(indentation+"temp3 = Norm_array[x3, y3, z3]\n")
                            else:
                                x3, y3, z3 = "0", "0", "0"
                            if ld != 0:
                                x4, y4, z4 = "x4", "y4", "z4"
                                S_file.write(indentation+"for x4 in range("+str(ld)+", -1, -1):\n")
                                S_file.write(indentation+"    for y4 in range("+str(ld)+"-x4, -1, -1):\n")
                                S_file.write(indentation+"        for z4 in range("+str(ld)+"-x4-y4, "+str(ld-1)+"-x4-y4, -1):\n")
                                indentation = indentation + "            "
                            else:
                                x4, y4, z4 = "0", "0", "0"
                                
                            if lc == 0 and ld == 0:
                                S_file.write(indentation+"primitives_buffer[i,j,k,l,counter] = ERI_expansion_coeff_sum_X_X_S_S(E_buff_1[i,j,"+x1+","+x2+",:,0],E_buff_1[i,j,"+y1+","+y2+",:,1],E_buff_1[i,j,"+z1+","+z2+",:,2],E_buff_2[k,l,"+x3+","+x4+",:,0],E_buff_2[k,l,"+y3+","+y4+",:,1],E_buff_2[k,l,"+z3+","+z4+",:,2],R_array,")
                            else:
                                S_file.write(indentation+"primitives_buffer[i,j,k,l,counter] = ERI_expansion_coeff_sum(E_buff_1[i,j,"+x1+","+x2+",:,0],E_buff_1[i,j,"+y1+","+y2+",:,1],E_buff_1[i,j,"+z1+","+z2+",:,2],E_buff_2[k,l,"+x3+","+x4+",:,0],E_buff_2[k,l,"+y3+","+y4+",:,1],E_buff_2[k,l,"+z3+","+z4+",:,2],R_array,")
<<<<<<< HEAD
                            # Just an ugly way to make the generated code abit nice, since x y z are always zero if angular moment is 0.
=======
                            # Just an ugly way to make the generated code a bit nice, since x y z are always zero if angular moment is 0.
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
                            if x1 == "x1":
                                S_file.write("x1+")
                            if x2 == "x2":
                                S_file.write("x2+")
                            S_file.write("1,")
                            if y1 == "y1":
                                S_file.write("y1+")
                            if y2 == "y2":
                                S_file.write("y2+")
                            S_file.write("1,")
                            if z1 == "z1":
                                S_file.write("z1+")
                            if z2 == "z2":
                                S_file.write("z2+")
                            S_file.write("1,")
                            if x3 == "x3":
                                S_file.write("x3+")
                            if x4 == "x4":
                                S_file.write("x4+")
                            S_file.write("1,")
                            if y3 == "y3":
                                S_file.write("y3+")
                            if y4 == "y4":
                                S_file.write("y4+")
                            S_file.write("1,")
                            if z3 == "z3":
                                S_file.write("z3+")
                            if z4 == "z4":
                                S_file.write("z4+")
                            S_file.write("1)")

                            if lc > 1:
                                S_file.write("*temp3")
                            elif lb > 1:
                                S_file.write("*temp2")
                            elif la > 1:
                                S_file.write("*temp1")
                            if ld > 1:
                                S_file.write("*Norm_array[x4, y4, z4]")
                            S_file.write("\n")
                            S_file.write(indentation+"counter += 1\n")
                            S_file.write("                    primitives_buffer[i,j,k,l,:"+str(combinations)+"] = 1.0/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:"+str(combinations)+"]\n")
                            S_file.write("    for i in range(0, "+str(combinations)+"):\n")
                            S_file.write("        output_buffer[i] = Contraction_two_electron(primitives_buffer[:,:,:,:,i], Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4)")
                            """
                            S_file.write("    output_buffer[:"+str(combinations)+"] = 0.0\n")
                            S_file.write("    for i in range(number_primitive_1):\n")
                            S_file.write("        temp1 = Contra_coeffs_1[i]\n")
                            S_file.write("        for j in range(number_primitive_2):\n")
                            S_file.write("            temp2 = temp1*Contra_coeffs_2[j]\n")
                            S_file.write("            for k in range(number_primitive_3):\n")
                            S_file.write("               temp3 = temp2*Contra_coeffs_3[k]\n")
                            S_file.write("               for l in range(number_primitive_4):\n")
                            S_file.write("                   temp4 = temp3*Contra_coeffs_4[l]\n")
                            S_file.write("                   for m in range("+str(combinations)+"):\n")
                            S_file.write("                       output_buffer[m] += primitives_buffer[i,j,k,l,m]*temp4\n")
                            S_file.write("    return output_buffer")
                            """
                            # For angular moment 0 and 1, the second part of the normalization is the 
                            #  same for all of the pritimives. Therefore it can be applied after contraction.
                            pi = 3.141592653589793238462643383279
                            total_same_norm = 2.0*pi**(5.0/2.0)  # This is the 2*pi**(5/2) factor on the integrals, applied after contraction
                            # Multiplying the normalization to the factor, to end up with one number
                            if la == 0:
                                total_same_norm *= PsuedoNorm2(0,0,0)
                            elif la == 1:
                                total_same_norm *= PsuedoNorm2(1,0,0)
                            if lb == 0:
                                total_same_norm *= PsuedoNorm2(0,0,0)
                            elif lb == 1:
                                total_same_norm *= PsuedoNorm2(1,0,0)
                            if lc == 0:
                                total_same_norm *= PsuedoNorm2(0,0,0)
                            elif lc == 1:
                                total_same_norm *= PsuedoNorm2(1,0,0)
                            if ld == 0:
                                total_same_norm *= PsuedoNorm2(0,0,0)
                            elif ld == 1:
                                total_same_norm *= PsuedoNorm2(1,0,0)
                            S_file.write("*"+str(total_same_norm))
                            S_file.write("\n")
                            S_file.write("    return output_buffer\n")
                            S_file.write("\n\n")
