import numpy


def E(i, j, t):
    output = "E["+str(i)+","+str(j)+","+str(t)+",:] = "
    return_check_i = [0,0,0]
    return_check_j = [0,0,0]
    if (t < 0) or (t > (i + j)):
        output += "0"
    elif i == j == t == 0:
        output += "np.exp(-q*XAB*XAB)"
    elif j == 0:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "p12 * E["+str(i-1)+","+str(j)+","+str(t-1)+",:]"
            return_check_i[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPA * E["+str(i-1)+","+str(j)+","+str(t)+",:]"
            return_check_i[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            if t+1 == 1:
                output += "E["+str(i-1)+","+str(j)+","+str(t+1)+",:]"
            else:
                output += str(t+1.0)+" * E["+str(i-1)+","+str(j)+","+str(t+1)+",:]"
            return_check_i[2] = 1
    else:
        if not ((t-1 < 0) or (t-1 > i-1 + j)):
            output += "p12 * E["+str(i)+","+str(j-1)+","+str(t-1)+",:]"
            return_check_j[0] = 1
        if not ((t < 0) or (t > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            output += "XPB * E["+str(i)+","+str(j-1)+","+str(t)+",:]"
            return_check_j[1] = 1
        if not ((t+1 < 0) or (t+1 > i-1 + j)):
            if output != "E["+str(i)+","+str(j)+","+str(t)+",:] = ":
                output += " + "
            if t+1 == 1:
                output += "E["+str(i)+","+str(j-1)+","+str(t+1)+",:]"
            else:
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


def generate_expansion_coefficients(max_angular_moment):
    global steps
    S_file = open("slowquant/molecularintegrals/expansion_coefficients.py", "w+")
    S_file.write("import numpy as np\n")
<<<<<<< HEAD
    S_file.write("from numba import jit, float64\n")
=======
    #S_file.write("from numba import jit, float64\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
    S_file.write("\n\n")
    
    for la in range(max_angular_moment+1):
        for lb in range(max_angular_moment+1):
            if la >= lb and la+lb <= 2*max_angular_moment:
                for lc in range(0, la+lb+1):
                    if lc == la+lb or lc == 0:
<<<<<<< HEAD
                        S_file.write("@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)\n")
=======
                        #S_file.write("@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
                        S_file.write("def E_"+str(la)+"_"+str(lb)+"_"+str(lc)+"(q, p12, XAB, XPA, XPB, E):\n")
                        steps = []
                        for i in range(la, -1, -1):
                            for j in range(lb, -1, -1):
                                for t in range(lc+1):
                                    if  i <= la and j <= lb and i+j >= t and i+j <= la+lb:
                                        E(i, j, t)
                        unique_steps = []
                        for k in range(-1, -len(steps)-1, -1):
                            if steps[k] not in unique_steps:
                                unique_steps.append(steps[k])
                        for step in unique_steps:
                            S_file.write("    "+step+"\n")
                        S_file.write("    return E\n")
                        S_file.write("\n\n")
        for lb in range(la+1, la+1+2): # kinetic energy integrals
            if la+lb <= 2*max_angular_moment + 2:
                for lc in range(0, la+lb+1): # keeping the lc part incase needed for future integrals
                    if lc == 0:
<<<<<<< HEAD
                        S_file.write("@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)\n")
=======
                        #S_file.write("@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)\n")
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
                        S_file.write("def E_"+str(la)+"_"+str(lb)+"_"+str(lc)+"(q, p12, XAB, XPA, XPB, E):\n")
                        steps = []
                        for i in range(la, -1, -1):
                            for j in range(lb, -1, -1):
                                for t in range(lc+1):
                                    if  i <= la and j <= lb and i+j >= t and i+j <= la+lb:
                                        E(i, j, t)
                        unique_steps = []
                        for k in range(-1, -len(steps)-1, -1):
                            if steps[k] not in unique_steps:
                                unique_steps.append(steps[k])
                        for step in unique_steps:
                            S_file.write("    "+step+"\n")
                        S_file.write("    return E\n")
                        S_file.write("\n\n")
                                            