import numpy as np


def R(t, u, v, n):
    return_check_v = [0,0]
    return_check_u = [0,0]
    return_check_t = [0,0]
    if n == 0:
        output = "R["+str(t)+","+str(u)+","+str(v)+"] = "
    else:
        output = "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = "
    if t == 0 and u == 0 and v == 0:
        if n == 0:
            output += "boys_function_n_zero(p*RPC*RPC)"
        elif n == 1:
            output += "-2.0*p * boys_function("+str(n)+",p*RPC*RPC)"
        else:
            output += "(-2.0*p)**"+str(n)+".0 * boys_function("+str(n)+",p*RPC*RPC)"
    elif t == 0 and u == 0:
        if v > 1:
            if v-1 == 1:
                output += "R_"+str(t)+"_"+str(u)+"_"+str(v-2)+"_"+str(n+1)
            else:
                if v-1 > 1:
                    output += str(v-1)+".0 * R_"+str(t)+"_"+str(u)+"_"+str(v-2)+"_"+str(n+1)
                else:
                    output += "R_"+str(t)+"_"+str(u)+"_"+str(v-2)+"_"+str(n+1)
            return_check_v[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+"] = " and output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "ZPC * R_"+str(t)+"_"+str(u)+"_"+str(v-1)+"_"+str(n+1)
        return_check_v[1] = 1
    elif t == 0:
        if u > 1:
            if u-1 == 1:
                output += "R_"+str(t)+"_"+str(u-2)+"_"+str(v)+"_"+str(n+1)
            else:
                if u-1 > 1:
                    output += str(u-1)+".0 * R_"+str(t)+"_"+str(u-2)+"_"+str(v)+"_"+str(n+1)
                else:
                    output += "R_"+str(t)+"_"+str(u-2)+"_"+str(v)+"_"+str(n+1)
            return_check_u[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+"] = " and output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "YPC * R_"+str(t)+"_"+str(u-1)+"_"+str(v)+"_"+str(n+1)
        return_check_u[1] = 1
    else:
        if t > 1:
            if t-1 == 0:
                output += "R_"+str(t-2)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
            else:
                if t-1 > 1:
                    output += str(t-1)+".0 * R_"+str(t-2)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
                else:
                    output += "R_"+str(t-2)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
            return_check_t[0] = 1
        if output != "R["+str(t)+","+str(u)+","+str(v)+"] = " and output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "XPC * R_"+str(t-1)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
        return_check_t[1] = 1
    
    if output != "R["+str(t)+","+str(u)+","+str(v)+"] = " and output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
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
        
        
def write_hermite_integral(max_angular):
    global steps
    S_file = open("slowquant/molecularintegrals/hermite_integral.py", "w+")
    S_file.write("import numpy as np\n")
    S_file.write("from numba import jit, float64\n")
    S_file.write("from slowquant.molecularintegrals.utility import boys_function, boys_function_n_zero\n")
    S_file.write("\n\n")
    
    for la in range(max_angular+1):
        for lb in range(max_angular+1):
            if la >= lb:
                for lc in range(max_angular+1):
                    for ld in range(max_angular+1):
                        if lc >= ld and la*(la+1)//2+lb >= lc*(lc+1)//2+ld:
                            S_file.write("@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)\n")
                            S_file.write("def R_"+str(la)+"_"+str(lb)+"_"+str(lc)+"_"+str(ld)+"(p, XPC, YPC, ZPC, RPC, R):\n")
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
                                   
                            
                            for i in range(0, len(unique_steps)):
                                if "[" not in unique_steps[i]:
                                    right_side = unique_steps[i].split(" = ")[0]
                                    left_side = unique_steps[i].split(" = ")[1]
                                    counter = 0
                                    for j in unique_steps:
                                        if right_side in j.split(" = ")[1]:
                                            counter += 1
                                            if counter > 1:
                                                break
                                    if counter == 1:
                                        for j in range(0 ,len(unique_steps)):
                                            if right_side in unique_steps[j].split(" = ")[1]:
                                                unique_steps[j] = unique_steps[j].replace(right_side, "("+left_side+")")
                                                break

                            term_dict = {}
                            for i in unique_steps:
                                term = i.split(" = ")[0]
                                if "[" not in i:
                                    if term not in term_dict:
                                        term_dict[term] = 0
                            for term in term_dict:
                                for i in unique_steps:
                                    if term in i.split(" = ")[1]:
                                        term_dict[term] += 1
                            new_list = []
                            for i in unique_steps:
                                if "["  in i:
                                    new_list.append(i)
                                elif term_dict[i.split(" = ")[0]] != 0:
                                    if term_dict[i.split(" = ")[0]] == 1:
                                        print("Oh no")
                                    new_list.append(i)
                                    
                                    
                            for step in new_list:
                                S_file.write("    "+step+"\n")
                            S_file.write("    return R[:,:,:,0]\n")
                            S_file.write("\n\n")
