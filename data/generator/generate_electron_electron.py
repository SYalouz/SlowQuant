import numpy


def R(t, u, v, n):
    return_check_v = [0,0]
    return_check_u = [0,0]
    return_check_t = [0,0]
    output = "R["+str(t)+","+str(u)+","+str(v)+","+str(n)+"] = "
    if t == 0 and u == 0 and v == 0:
        output += "(-2.0*alpha)**"+str(n)+" * boys_function("+str(n)+",alpha*RPC*RPC)"
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


def E_left(i, j, t):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    output = "E_left["+str(i)+","+str(j)+","+str(t)+",:] = "
    return_check_i = [0,0,0]
    return_check_j = [0,0,0]
    if (t < 0) or (t > (i + j)):
        output += "0"
    elif i == j == t == 0:
        output += "np.exp(-q_left*XAB_left*XAB_left)"
    elif j == 0:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p_left)) * E_left["+str(i-1)+","+str(j)+","+str(t-1)+",:]"
            return_check_i[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_left["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPA_left * E_left["+str(i-1)+","+str(j)+","+str(t)+",:]"
            return_check_i[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_left["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E_left["+str(i-1)+","+str(j)+","+str(t+1)+",:]"
            return_check_i[2] = 1
    else:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p_left)) * E_left["+str(i)+","+str(j-1)+","+str(t-1)+",:]"
            return_check_j[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_left["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPB_left * E_left["+str(i)+","+str(j-1)+","+str(t)+",:]"
            return_check_j[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_left["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E_left["+str(i)+","+str(j-1)+","+str(t+1)+",:]"
            return_check_j[2] = 1
  
    if output != "E_left["+str(i)+","+str(j)+","+str(t)+",:] = ":
        steps.append(output)
    if return_check_i == [1,0,0]:
        return E_left(i-1,j,t-1)
    elif return_check_i == [0,1,0]:
        return E_left(i-1,j,t)
    elif return_check_i == [0,0,1]:
        return E_left(i-1,j,t+1)
    elif return_check_i == [1,1,0]:
        return E_left(i-1,j,t-1) + E_left(i-1,j,t)
    elif return_check_i == [1,0,1]:
        return E_left(i-1,j,t-1) + E_left(i-1,j,t+1)
    elif return_check_i == [0,1,1]:
        return E_left(i-1,j,t) + E_left(i-1,j,t+1)
    elif return_check_i == [1,1,1]:
        return E_left(i-1,j,t-1) + E_left(i-1,j,t) + E_left(i-1,j,t+1)
    elif return_check_j == [1,0,0]:
        return E_left(i,j-1,t-1)
    elif return_check_j == [0,1,0]:
        return E_left(i,j-1,t)
    elif return_check_j == [0,0,1]:
        return E_left(i,j-1,t+1)
    elif return_check_j == [1,1,0]:
        return E_left(i,j-1,t-1) + E_left(i,j-1,t)
    elif return_check_j == [1,0,1]:
        return E_left(i,j-1,t-1) + E_left(i,j-1,t+1)
    elif return_check_j == [0,1,1]:
        return E_left(i,j-1,t) + E_left(i,j-1,t+1)
    elif return_check_j == [1,1,1]:
        return E_left(i,j-1,t-1) + E_left(i,j-1,t) + E_left(i,j-1,t+1)
    else:
        return 0.0
        
def E_right(i, j, t):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    output = "E_right["+str(i)+","+str(j)+","+str(t)+",:] = "
    return_check_i = [0,0,0]
    return_check_j = [0,0,0]
    if (t < 0) or (t > (i + j)):
        output += "0"
    elif i == j == t == 0:
        output += "np.exp(-q_right*XAB_right*XAB_right)"
    elif j == 0:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p_right)) * E_right["+str(i-1)+","+str(j)+","+str(t-1)+",:]"
            return_check_i[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_right["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPA_right * E_right["+str(i-1)+","+str(j)+","+str(t)+",:]"
            return_check_i[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_right["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E_right["+str(i-1)+","+str(j)+","+str(t+1)+",:]"
            return_check_i[2] = 1
    else:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "(1.0/(2.0*p_right)) * E_right["+str(i)+","+str(j-1)+","+str(t-1)+",:]"
            return_check_j[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E_right["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPB_right * E_right["+str(i)+","+str(j-1)+","+str(t)+",:]"
            return_check_j[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E_right["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += str(t+1.0)+" * E_right["+str(i)+","+str(j-1)+","+str(t+1)+",:]"
            return_check_j[2] = 1
  
    if output != "E_right["+str(i)+","+str(j)+","+str(t)+",:] = ":
        steps.append(output)
    if return_check_i == [1,0,0]:
        return E_right(i-1,j,t-1)
    elif return_check_i == [0,1,0]:
        return E_right(i-1,j,t)
    elif return_check_i == [0,0,1]:
        return E_right(i-1,j,t+1)
    elif return_check_i == [1,1,0]:
        return E_right(i-1,j,t-1) + E_right(i-1,j,t)
    elif return_check_i == [1,0,1]:
        return E_right(i-1,j,t-1) + E_right(i-1,j,t+1)
    elif return_check_i == [0,1,1]:
        return E_right(i-1,j,t) + E_right(i-1,j,t+1)
    elif return_check_i == [1,1,1]:
        return E_right(i-1,j,t-1) + E_right(i-1,j,t) + E_right(i-1,j,t+1)
    elif return_check_j == [1,0,0]:
        return E_right(i,j-1,t-1)
    elif return_check_j == [0,1,0]:
        return E_right(i,j-1,t)
    elif return_check_j == [0,0,1]:
        return E_right(i,j-1,t+1)
    elif return_check_j == [1,1,0]:
        return E_right(i,j-1,t-1) + E_right(i,j-1,t)
    elif return_check_j == [1,0,1]:
        return E_right(i,j-1,t-1) + E_right(i,j-1,t+1)
    elif return_check_j == [0,1,1]:
        return E_right(i,j-1,t) + E_right(i,j-1,t+1)
    elif return_check_j == [1,1,1]:
        return E_right(i,j-1,t-1) + E_right(i,j-1,t) + E_right(i,j-1,t+1)
    else:
        return 0.0
        
        
def write_electron_electron(max_angular):        
    S_file = open("../../slowquant/molecularintegrals/electron_electron.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("import math\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.utility import boys_function, ERI_expansion_coeff_sum\n")
    S_file.write("\n\n")
    
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                for lc in range(max_angular+1):
                    for ld in range(max_angular+1):
                        if la == 0 and lb == 0 and lc == 0 and ld == 0:
                            S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:]), nopython=True, cache=True)\n")
                            S_file.write("def electron_electron_integral_0_0_0_0(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer):\n")
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
                            S_file.write("            E1 = np.exp(-q_left*XAB_left[0]*XAB_left[0])*np.exp(-q_left*XAB_left[10]*XAB_left[1])*np.exp(-q_left*XAB_left[2]*XAB_left[2])\n")
                            S_file.write("\n")
                            S_file.write("            for k in range(0, number_primitive_3):\n")
                            S_file.write("                gauss_exp_1_right = gauss_exp_3[k]\n")
                            S_file.write("                for l in range(0, number_primitive_4):\n")
                            S_file.write("                    gauss_exp_2_right = gauss_exp_4[l]\n")
                            S_file.write("                    p_right = gauss_exp_1_right + gauss_exp_2_right\n")
                            S_file.write("                    q_right = gauss_exp_1_right * gauss_exp_2_right / p_right\n")
                            S_file.write("                    P_right = (gauss_exp_1_right*Coord_3 + gauss_exp_2_right*Coord_4) / p_right\n")
                            S_file.write("                    alpha = p_left*p_right/(p_left+p_right)\n")
                            S_file.write("                    XPC, YPC, ZPC = P_left - P_right\n")
                            S_file.write("                    RPC = ((XPC)**2+(YPC)**2+(ZPC)**2)**0.5\n")
                            S_file.write("\n")
                            S_file.write("                    if RPC == 0:\n")
                            S_file.write("                        primitives_buffer[i,j,k,l,0] = pi52/(p_left*p_right*(p_left+p_right)**0.5)\n")
                            S_file.write("                    else:\n")
                            S_file.write("                        primitives_buffer[i,j,k,l,0] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*E1*np.exp(-q_right*XAB_right[0]*XAB_right[0])*np.exp(-q_right*XAB_right[1]*XAB_right[1])*np.exp(-q_right*XAB_right[2]*XAB_right[2])*(pi/(4*alpha*RPC*RPC))**0.5*math.erf(alpha*RPC*RPC)\n")
                            S_file.write("\n")
                            S_file.write("    for i in range(0, number_primitive_1):\n")
                            S_file.write("        for j in range(0, number_primitive_2):\n")
                            S_file.write("            for k in range(0, number_primitive_3):\n")
                            S_file.write("                for l in range(0, number_primitive_4):\n")
                            S_file.write("                    output_buffer[0] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,0]\n")
                            S_file.write("    return output_buffer\n")
                            S_file.write("\n\n")
                        elif lc >= ld and la*(la+1)//2+lb >= lc*(lc+1)//2+ld:
                            S_file.write("@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:,:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:], float64[:], float64[:,:,:]), nopython=True, cache=True)\n")
                            S_file.write("def electron_electron_integral_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"(Coord_1, Coord_2, Coord_3, Coord_4, gauss_exp_1, gauss_exp_2, gauss_exp_3, gauss_exp_4, Contra_coeffs_1, Contra_coeffs_2, Contra_coeffs_3, Contra_coeffs_4, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, Contraction_3_buffer, Contraction_4_buffer, E_left, E_right, R, primitives_buffer_2, output_buffer, norm_array):\n")
                            S_file.write("    number_primitive_1 = gauss_exp_1.shape[0]\n")
                            S_file.write("    number_primitive_2 = gauss_exp_2.shape[0]\n")
                            S_file.write("    number_primitive_3 = gauss_exp_3.shape[0]\n")
                            S_file.write("    number_primitive_4 = gauss_exp_4.shape[0]\n")
                            combinations = (la+1)*(lb+1)*(lc+1)*(ld+1)*(la+2)*(lb+2)*(lc+2)*(ld+2)/16
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
                            global steps
                            steps = []
                            unique_steps = []
                            for i in range(la+lb, -1, -1):
                                for j in range(la+lb, -1, -1):
                                    if  i <= la and j <= lb:
                                        for t in range(0, i+j+1):
                                            E_left(i, j, t)
                            for i in range(-1, -len(steps)-1, -1):
                                if steps[i] not in unique_steps:
                                    unique_steps.append(steps[i])
                            for step in unique_steps:
                                S_file.write("            "+step+"\n")
                            S_file.write("\n")
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
                            S_file.write("                    \n")
                            steps = []
                            unique_steps = []
                            for i in range(lc+ld, -1, -1):
                                for j in range(lc+ld, -1, -1):
                                    if  i <= lc and j <= ld:
                                        for t in range(0, i+j+1):   
                                            E_right(i, j, t)
                            for i in range(-1, -len(steps)-1, -1):
                                if steps[i] not in unique_steps:
                                    unique_steps.append(steps[i])
                            for step in unique_steps:
                                S_file.write("                    "+step+"\n")
                            S_file.write("\n")
                            steps = []
                            unique_steps = []
                            for i in range(la+lb+lc+ld, -1, -1):
                                for j in range(la+lb+lc+ld, -1, -1):
                                    for k in range(la+lb+lc+ld, -1, -1):
                                        if i+j+k <= la+lb+lc+ld:
                                            R(i, j, k, 0)
                            for i in range(-1, -len(steps)-1, -1):
                                if steps[i] not in unique_steps:
                                    unique_steps.append(steps[i])
                            for step in unique_steps:
                                S_file.write("                    "+step+"\n")
                            S_file.write("\n")
                            counter = 0
                            for x1 in range(max_angular,-1, -1):
                                for y1 in range(max_angular,-1, -1):
                                    for z1 in range(max_angular,-1, -1):
                                        for x2 in range(max_angular,-1, -1):
                                            for y2 in range(max_angular,-1, -1):
                                                for z2 in range(max_angular,-1, -1):
                                                    for x3 in range(max_angular,-1, -1):
                                                        for y3 in range(max_angular,-1, -1):
                                                            for z3 in range(max_angular,-1, -1):
                                                                for x4 in range(max_angular,-1, -1):
                                                                    for y4 in range(max_angular,-1, -1):
                                                                        for z4 in range(max_angular,-1, -1):
                                                                            if x1+y1+z1 == la and x2+y2+z2 == lb and x3+y3+z3 == lc and x4+y4+z4 == ld:
                                                                                S_file.write("                    primitives_buffer[i,j,k,l,"+str(counter)+"] = ERI_expansion_coeff_sum(E_left["+str(x1)+","+str(x2)+",:,0],")
                                                                                S_file.write("E_left["+str(y1)+","+str(y2)+",:,1],")
                                                                                S_file.write("E_left["+str(z1)+","+str(z2)+",:,2],")
                                                                                S_file.write("E_right["+str(x3)+","+str(x4)+",:,0],")
                                                                                S_file.write("E_right["+str(y3)+","+str(y4)+",:,1],")
                                                                                S_file.write("E_right["+str(z3)+","+str(z4)+",:,2],")
                                                                                S_file.write("R[:,:,:,0],"+str(x1+x2+1)+","+str(y1+y2+1)+","+str(z1+z2+1)+","+str(x3+x4+1)+","+str(y3+y4+1)+","+str(z3+z4+1)+")")
                                                                                S_file.write("*norm_array["+str(x1)+","+str(y1)+","+str(z1)+"]*norm_array["+str(x2)+","+str(y2)+","+str(z2)+"]*norm_array["+str(x3)+","+str(y3)+","+str(z3)+"]*norm_array["+str(x4)+","+str(y4)+","+str(z4)+"]\n")
                                                                                counter += 1
                            S_file.write("\n")
                            S_file.write("                    primitives_buffer[i,j,k,l,:] = pi52/(p_left*p_right*(p_left+p_right)**0.5)*primitives_buffer[i,j,k,l,:]\n")
                            S_file.write("\n")
                            S_file.write("    for out in range(0, "+str(int(combinations))+"):\n")
                            S_file.write("        for i in range(0, number_primitive_1):\n")
                            S_file.write("            for j in range(0, number_primitive_2):\n")
                            S_file.write("                for k in range(0, number_primitive_3):\n")
                            S_file.write("                    for l in range(0, number_primitive_4):\n")
                            S_file.write("                        output_buffer[out] += Contraction_1_buffer[i]*Contraction_2_buffer[j]*Contraction_3_buffer[k]*Contraction_4_buffer[l]*primitives_buffer[i,j,k,l,out]\n")
                            S_file.write("    return output_buffer\n")
                            S_file.write("\n\n")


write_electron_electron(1)




























