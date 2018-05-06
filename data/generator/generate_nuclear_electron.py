import numpy


def R(t, u, v, n):
    return_check_v = [0,0]
    return_check_u = [0,0]
    return_check_t = [0,0]
    output = "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = "
    if t == 0 and u == 0 and v == 0:
        output += "(-2.0*p)**"+str(n)+" * boys_function("+str(n)+",p*RPC*RPC)"
    elif t == 0 and u == 0:
        if v > 1:
            output += str(v-1)+" * R["+str(t)+","+str(u)+","+str(v-2)+","+str(n+1)+"]"
            return_check_v[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = ":
            output += " + "
        output += "ZPC * R["+str(t)+","+str(u)+","+str(v-1)+","+str(n+1)+"]"
        return_check_v[1] = 1
    elif t == 0:
        if u > 1:
            output += str(u-1)+" * R["+str(t)+","+str(u-2)+","+str(v)+","+str(n+1)+"]"
            return_check_u[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = ":
            output += " + "
        output += "YPC * R["+str(t)+","+str(u-1)+","+str(v)+","+str(n+1)+"]"
        return_check_u[1] = 1
    else:
        if t > 1:
            output += str(t-1)+" * R["+str(t-2)+","+str(u)+","+str(v)+","+str(n+1)+"]"
            return_check_t[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = ":
            output += " + "
        output += "XPC * R["+str(t-1)+","+str(u)+","+str(v)+","+str(n+1)+"]"
        return_check_t[1] = 1
    
    if output != "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = ":
        steps.append(output)
    if return_check_v == [1,1]:
        return R(t,u,v-2,n+1) + R(t,u,v-1,n+1)
    elif return_check_v == [0,1]:
        return R(t,u,v-1,n+1)
    elif return_check_u == [1,1]:
        return R(t,u-2,v,n+1) + R(t,u-1,v,n+1)
    elif return_check_u == [0,1]:
        return R(t,u-1,v,n+1)
    elif return_check_t == [1,1]:
        return R(t-2,u,v,n+1) + R(t-1,u,v,n+1)
    elif return_check_t == [0,1]:
        return R(t-1,u,v,n+1)
    else:
        return 0.0


def E(i, j, t):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    output = "E["+str(i)+","+str(j)+","+str(t)+",:] = "
    return_check_i = [0,0,0]
    return_check_j = [0,0,0]
    if (t < 0) or (t > (i + j)):
        output += "0"
    elif i == j == t == 0:
        output += "np.exp(-q*XAB*XAB)"
    elif j == 0:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p)) * E["+str(i-1)+","+str(j)+","+str(t-1)+",:]"
            return_check_i[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPA * E["+str(i-1)+","+str(j)+","+str(t)+",:]"
            return_check_i[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E["+str(i-1)+","+str(j)+","+str(t+1)+",:]"
            return_check_i[2] = 1
    else:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p)) * E["+str(i)+","+str(j-1)+","+str(t-1)+",:]"
            return_check_j[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPB * E["+str(i)+","+str(j-1)+","+str(t)+",:]"
            return_check_j[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E["+str(i)+","+str(j-1)+","+str(t+1)+",:]"
            return_check_j[2] = 1
  
    if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
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

        
def write_nuclear_electron(max_angular):        
    S_file = open("../../slowquant/molecularintegrals/nuclear_electron_potential.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.utility import Normalization, boys_function\n")
    S_file.write("\n\n")
    
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                global steps
                steps = []
                for i in range(la+lb, -1, -1):
                    for j in range(la+lb, -1, -1):
                        if  i+j <= la+lb:
                            for t in range(0, i+j+1):
                                E(i, j, t)
                unique_steps = []
                for i in range(-1, -len(steps)-1, -1):
                    if steps[i] not in unique_steps:
                        unique_steps.append(steps[i])
                S_file.write("@jit(float64[:](float64[:], float64[:], float64, float64, float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)\n")
                S_file.write("def primitive_nuclear_electron_potential_"+str(la)+"_"+str(lb)+"(Coord_1, Coord_2, guass_exp_1, gauss_exp_2, atoms, E, R, primitive):\n")
                S_file.write("    pi = 3.141592653589793238462643383279\n")
                S_file.write("    p = guass_exp_1 + gauss_exp_2\n")
                S_file.write("    q = guass_exp_1 * gauss_exp_2 / p\n")
                S_file.write("    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p\n")
                S_file.write("    XAB = Coord_1 - Coord_2\n")
                S_file.write("    XPA = P - Coord_1\n")
                S_file.write("    XPB = P - Coord_2\n")
                S_file.write("    primitive[:] = 0.0\n")
                S_file.write("    \n")
                for step in unique_steps:
                    S_file.write("    "+step+"\n")
                S_file.write("\n")
                S_file.write("    for i in range(0, len(atoms)):\n")
                S_file.write("        XPC, YPC, ZPC = P - atoms[i,1:4]\n")
                S_file.write("        RPC = (XPC**2 + YPC**2 + ZPC**2)**0.5\n")
                S_file.write("        Charge = atoms[i,0]\n")
                steps = []
                unique_steps = []
                for i in range(la+lb, -1, -1):
                    for j in range(la+lb, -1, -1):
                        for k in range(la+lb, -1, -1):
                            if i+j+k <= la+lb:
                                R(i, j, k, 0)
                for i in range(-1, -len(steps)-1, -1):
                    if steps[i] not in unique_steps:
                        unique_steps.append(steps[i])
                for step in unique_steps:
                    S_file.write("        "+step+"\n")
                S_file.write("\n")
                counter = 0
                
                for i in range(max_angular,-1, -1):
                    for k in range(max_angular,-1, -1):
                        for m in range(max_angular,-1, -1):
                            for j in range(max_angular,-1, -1):
                                for l in range(max_angular,-1, -1):
                                    for n in range(max_angular,-1, -1):
                                        if i+k+m == la and j+l+n == lb:
                                            S_file.write("        for t in range(0, "+str(i+j+1)+"):\n")
                                            S_file.write("            for u in range(0, "+str(k+l+1)+"):\n")
                                            S_file.write("                for v in range(0, "+str(m+n+1)+"):\n")
                                            S_file.write("                    primitive["+str(counter)+"] += Charge*E["+str(i)+","+str(j)+",t,0]*E["+str(k)+","+str(l)+",u,1]*E["+str(m)+","+str(n)+",v,2]*R[t,u,v,0]\n")
                                            """
                                            S_file.write("        primitive["+str(counter)+"] += Charge*np.dot(np.dot(np.dot(R[:"+str(i+j+1)+",:"+str(k+l+1)+",:"+str(m+n+1)+",0],E["+str(m)+","+str(n)+",:"+str(m+n+1)+",2]),E["+str(k)+","+str(l)+",:"+str(k+l+1)+",1]),E["+str(i)+","+str(j)+",:"+str(i+j+1)+",0])\n")
                                            """
                                            counter += 1
                S_file.write("\n")
                S_file.write("    return -2.0*pi/p*primitive\n")
                S_file.write("\n\n")
                
                
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:], float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)\n")
                S_file.write("def nuclear_electron_integral_"+str(la)+"_"+str(lb)+"(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, atoms, E_buffer, R_buffer, primitives_buffer_2):\n")
                S_file.write("    number_primitive_1 = len(gauss_exp_1)\n")
                S_file.write("    number_primitive_2 = len(gauss_exp_2)\n")
                S_file.write("    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]\n")
                S_file.write("    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]\n")
                if la == 0:
                    S_file.write("    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:1]\n")
                    S_file.write("    for i in range(0, len(gauss_exp_1)):\n")
                    S_file.write("        for j in range(0, len(gauss_exp_2)):\n")
                    S_file.write("            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_0_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)\n")
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
                    S_file.write("            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_1_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)\n")
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
                    S_file.write("            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_1_1(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)\n")
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
    write_nuclear_electron(1)
        