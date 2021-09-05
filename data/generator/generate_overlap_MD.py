import numpy as np

def write_overlap(max_angular):
    S_file = open("slowquant/molecularintegrals/overlap_MD.py", "w+")
    S_file.write("import numpy as np\n")
<<<<<<< HEAD
    S_file.write("from numba import jit, float64\n")
=======
    #S_file.write("from numba import jit, float64\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
    S_file.write("from slowquant.molecularintegrals.expansion_coefficients import *\n")
    S_file.write("from slowquant.molecularintegrals.utility import Contraction_one_electron")
    S_file.write("\n\n")
    for lb in range(max_angular+1):
        for la in range(lb, max_angular+1):
<<<<<<< HEAD
            S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:]), nopython=True, cache=True)\n")
=======
            #S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:,:]), nopython=True, cache=True)\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
            S_file.write("def overlap_integral_"+str(la)+"_"+str(lb)+"_MD(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Norm_array, E_buffer):\n")
            S_file.write("    number_primitive_1 = gauss_exp_1.shape[0]\n")
            S_file.write("    number_primitive_2 = gauss_exp_2.shape[0]\n")
            S_file.write("    pi = 3.141592653589793238462643383279\n")
            S_file.write("    XAB = Coord_1 - Coord_2\n")
            S_file.write("    for i in range(0, number_primitive_1):\n")
            S_file.write("        for j in range(0, number_primitive_2):\n")
            S_file.write("            p = gauss_exp_1[i] + gauss_exp_2[j]\n")
            S_file.write("            q = gauss_exp_1[i] * gauss_exp_2[j] / p\n")
            S_file.write("            P = (gauss_exp_1[i]*Coord_1 + gauss_exp_2[j]*Coord_2) / p\n")
            S_file.write("            XPA = P - Coord_1\n")
            S_file.write("            XPB = P - Coord_2\n")
            S_file.write("            p12 = 1.0/(2.0*p)\n")
            S_file.write("            E_buffer = E_"+str(la)+"_"+str(lb)+"_"+str(0)+"(q, p12, XAB, XPA, XPB, E_buffer)\n")
            S_file.write("            counter = 0\n")
            S_file.write("            for x1 in range("+str(la)+", -1, -1):\n")
            S_file.write("                for y1 in range("+str(la)+"-x1, -1, -1):\n")
            S_file.write("                    for z1 in range("+str(la)+"-x1-y1, "+str(la-1)+"-x1-y1, -1):\n")
            S_file.write("                        temp1 = Norm_array[x1, y1, z1]\n")
            S_file.write("                        for x2 in range("+str(lb)+", -1, -1):\n")
            S_file.write("                            for y2 in range("+str(lb)+"-x2, -1, -1):\n")
            S_file.write("                                for z2 in range("+str(lb)+"-x2-y2, "+str(lb-1)+"-x2-y2, -1):\n")
            S_file.write("                                   primitives_buffer[i,j,counter] = E_buffer[x1,x2,0,0]*E_buffer[y1,y2,0,1]*E_buffer[z1,z2,0,2]*temp1*Norm_array[x2,y2,z2]\n")
            S_file.write("                                   counter += 1\n")
            S_file.write("            primitives_buffer[i,j,:] = (pi/p)**(3.0/2.0)*primitives_buffer[i,j,:]\n")
            S_file.write("\n")
            S_file.write("    for i in range(0, "+str((la+1)*((la+1)+1)//2*(lb+1)*((lb+1)+1)//2)+"):\n")
            S_file.write("        output_buffer[i] = Contraction_one_electron(primitives_buffer[:,:,i], Contra_coeffs_1, Contra_coeffs_2)\n")
            S_file.write("    return output_buffer\n")
            S_file.write("\n\n")
