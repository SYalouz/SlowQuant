import numpy


def E(i, j, t):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    output = "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = "
    return_check_i = [0,0,0]
    return_check_j = [0,0,0]
    if (t < 0) or (t > (i + j)):
        output += "0"
    elif i == j == t == 0:
        output += "np.exp(-q*XAB*XAB)"
    elif j == 0:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p)) * E_"+str(i-1)+"_"+str(j)+"_"+str(t-1)
            return_check_i[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = ":
                output += " + "
            output += "XPA * E_"+str(i-1)+"_"+str(j)+"_"+str(t)
            return_check_i[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = ":
                output += " + "
            output += str(t+1.0)+" * E_"+str(i-1)+"_"+str(j)+"_"+str(t+1)
            return_check_i[2] = 1
    else:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p)) * E_"+str(i)+"_"+str(j-1)+"_"+str(t-1)
            return_check_j[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = ":
                output += " + "
            output += "XPB * E_"+str(i)+"_"+str(j-1)+"_"+str(t)
            return_check_j[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = ":
                output += " + "
            output += str(t+1.0)+" * E_"+str(i)+"_"+str(j-1)+"_"+str(t+1)
            return_check_j[2] = 1
  
    if output != "E_"+str(i)+"_"+str(j)+"_"+str(t)+" = ":
        steps.append(output)
    if return_check_i == [1,0,0]:
        return E(i-1,j,t-1)
    elif return_check_i == [0,1,0]:
        return E(i-1,j,t)
    elif return_check_i == [0,0,1]:
        return E(i-1,j,t+1)
    elif return_check_i == [1,1,0]:
        return E(i-1,j,t-1) + E(i-1,j,t)
    elif return_check_i == [1,0,1]:
        return E(i-1,j,t-1) + E(i-1,j,t+1)
    elif return_check_i == [0,1,1]:
        return E(i-1,j,t) + E(i-1,j,t+1)
    elif return_check_i == [1,1,1]:
        return E(i-1,j,t-1) + E(i-1,j,t) + E(i-1,j,t+1)
    elif return_check_j == [1,0,0]:
        return E(i,j-1,t-1)
    elif return_check_j == [0,1,0]:
        return E(i,j-1,t)
    elif return_check_j == [0,0,1]:
        return E(i,j-1,t+1)
    elif return_check_j == [1,1,0]:
        return E(i,j-1,t-1) + E(i,j-1,t)
    elif return_check_j == [1,0,1]:
        return E(i,j-1,t-1) + E(i,j-1,t+1)
    elif return_check_j == [0,1,1]:
        return E(i,j-1,t) + E(i,j-1,t+1)
    elif return_check_j == [1,1,1]:
        return E(i,j-1,t-1) + E(i,j-1,t) + E(i,j-1,t+1)
    else:
        return 0.0
        

def write_overlap(max_angular):
    S_file = open("../../slowquant/molecularintegrals/overlap.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.utility import Normalization\n")
    S_file.write("\n\n")
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                global steps
                steps = []
                for i in range(la, -1, -1):
                    for j in range(lb, -1, -1):
                        E(i, j, 0)
                unique_steps = []
                for i in range(-1, -len(steps)-1, -1):
                    if steps[i] not in unique_steps:
                        unique_steps.append(steps[i])
                    
                S_file.write("@jit(float64[:](float64[:], float64[:], float64, float64), nopython=True, cache=True)\n")
                S_file.write("def primitive_overlap_"+str(la)+"_"+str(lb)+"(Coord_1, Coord_2, guass_exp_1, gauss_exp_2):\n")
                S_file.write("    pi = 3.141592653589793238462643383279\n")
                S_file.write("    p = guass_exp_1 + gauss_exp_2\n")
                S_file.write("    q = guass_exp_1 * gauss_exp_2 / p\n")
                S_file.write("    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p\n")
                S_file.write("    XAB = Coord_1 - Coord_2\n")
                S_file.write("    XPA = P - Coord_1\n")
                S_file.write("    XPB = P - Coord_2\n")
                for step in unique_steps:
                    S_file.write("    "+step+"\n")
                S_file.write("\n")
                if la == 0:
                    S_file.write("    return np.array([(pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_0_0_0[2]])\n")
                elif la == 1 and lb == 0:
                    S_file.write("    return np.array([(pi/p)**(3/2) * E_1_0_0[0] * E_0_0_0[1] * E_0_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_0_0[1] * E_0_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_1_0_0[2]])\n")
                elif la == 1 and lb == 1:
                    S_file.write("    return np.array([(pi/p)**(3/2) * E_1_1_0[0] * E_0_0_0[1] * E_0_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_1_0_0[0] * E_0_1_0[1] * E_0_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_1_0_0[0] * E_0_0_0[1] * E_0_1_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_1_0[0] * E_1_0_0[1] * E_0_1_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_1_0[1] * E_0_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_1_0_0[1] * E_0_1_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_1_0[0] * E_0_0_0[1] * E_1_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_1_0[1] * E_1_0_0[2],\n")
                    S_file.write("                     (pi/p)**(3/2) * E_0_0_0[0] * E_0_0_0[1] * E_1_1_0[2]])\n")
                else:
                    S_file.write("    return None\n")
                S_file.write("\n\n")
                
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:]), nopython=True, cache=True)\n")
                S_file.write("def overlap_integral_"+str(la)+"_"+str(lb)+"(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer):\n")
                S_file.write("    number_primitive_1 = len(gauss_exp_1)\n")
                S_file.write("    number_primitive_2 = len(gauss_exp_2)\n")
                S_file.write("    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]\n")
                S_file.write("    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]\n")
                if la == 0:
                    S_file.write("    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:1]\n")
                    S_file.write("    for i in range(0, len(gauss_exp_1)):\n")
                    S_file.write("        for j in range(0, len(gauss_exp_2)):\n")
                    S_file.write("            primitives_buffer[i,j,:] = primitive_overlap_0_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_1)):\n")
                    S_file.write("        Contraction_1_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_2)):\n")
                    S_file.write("        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])\n")
                    S_file.write("    output_buffer[0] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    return output_buffer\n")
                    S_file.write("\n\n")
                elif la == 1 and lb == 0:
                    S_file.write("    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:3]\n")
                    S_file.write("    for i in range(0, len(gauss_exp_1)):\n")
                    S_file.write("        for j in range(0, len(gauss_exp_2)):\n")
                    S_file.write("            primitives_buffer[i,j,:] = primitive_overlap_1_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_1)):\n")
                    S_file.write("        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_2)):\n")
                    S_file.write("        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])\n")
                    S_file.write("    output_buffer[0] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])\n")
                    S_file.write("    output_buffer[1] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])\n")
                    S_file.write("    output_buffer[2] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    return output_buffer\n")
                    S_file.write("\n\n")
                elif la == 1 and lb == 1:
                    S_file.write("    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:9]\n")
                    S_file.write("    for i in range(0, len(gauss_exp_1)):\n")
                    S_file.write("        for j in range(0, len(gauss_exp_2)):\n")
                    S_file.write("            primitives_buffer[i,j,:] = primitive_overlap_1_1(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j])\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_1)):\n")
                    S_file.write("        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]\n")
                    S_file.write("    for i in range(0, len(Contra_coeffs_2)):\n")
                    S_file.write("        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])\n")
                    S_file.write("    output_buffer[0] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])\n")
                    S_file.write("    output_buffer[1] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])\n")
                    S_file.write("    output_buffer[2] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,3])\n")
                    S_file.write("    output_buffer[3] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,4])\n")
                    S_file.write("    output_buffer[4] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,5])\n")
                    S_file.write("    output_buffer[5] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,6])\n")
                    S_file.write("    output_buffer[6] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,7])\n")
                    S_file.write("    output_buffer[7] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,8])\n")
                    S_file.write("    output_buffer[8] = np.dot(temp, Contraction_2_buffer)\n")
                    S_file.write("    return output_buffer\n")
                    S_file.write("\n\n")
                elif la == 2 and lb == 0:
                    S_file.write("    return None\n")
                elif la == 2 and lb == 1:
                    S_file.write("    return None\n")
                elif la == 2 and lb == 2:
                    S_file.write("    return None\n")
    S_file.close()
            

if __name__ == "__main__":
    write_overlap(1)
