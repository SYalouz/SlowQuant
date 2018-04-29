import numpy


def R(t, u, v, n):
    return_check_v = [0,0]
    return_check_u = [0,0]
    return_check_t = [0,0]
    output = "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = "
    if t == 0 and u == 0 and v == 0:
        output += "(-2.0*p)**"+str(n)+" * boys_function("+str(n)+"*RPC2)"
    elif t == 0 and u == 0:
        if v > 1:
            output += str(v-1)+" * R_"+str(t)+"_"+str(u)+"_"+str(v-2)+"_"+str(n+1)
            return_check_v[0] = 1
        if output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "ZPC * R_"+str(t)+"_"+str(u)+"_"+str(v-1)+"_"+str(n+1)
        return_check_v[1] = 1
    elif t == 0:
        if u > 1:
            output += str(u-1)+" * R_"+str(t)+"_"+str(u-2)+"_"+str(v)+"_"+str(n+1)
            return_check_u[0] = 1
        if output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "YPC * R_"+str(t)+"_"+str(u-1)+"_"+str(v)+"_"+str(n+1)
        return_check_u[1] = 1
    else:
        if t > 1:
            output += str(t-1)+" * R_"+str(t-2)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
            return_check_t[0] = 1
        if output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
            output += " + "
        output += "XPC * R_"+str(t-1)+"_"+str(u)+"_"+str(v)+"_"+str(n+1)
        return_check_t[1] = 1
    
    if output != "R_"+str(t)+"_"+str(u)+"_"+str(v)+"_"+str(n)+" = ":
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
        
        
        
        
        
        
        
        



global steps
steps = []
la = 1
ma = 1
na = 1
for i in range(la, -1, -1):
    for j in range(la, -1, -1):
        for k in range(la, -1, -1):
            R(i, j, k, 0)
unique_steps = []
for i in range(-1, -len(steps)-1, -1):
    if steps[i] not in unique_steps:
        unique_steps.append(steps[i])
        
for i in unique_steps:
    print(i)

        