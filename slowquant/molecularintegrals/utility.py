import numpy as np
from numba import jit, float64, int64
from numba.types import Tuple


@jit(float64(float64),nopython=True,cache=True)
def factorial2(n):
    n_range = int(n)
    out = 1.0
    if n > 0:
        for i in range(0, int(n_range+1)//2):
            out = out*(n-2*i)
    return out


@jit(float64(float64,float64),nopython=True,cache=True)
def boys_function(m,z):
    pi = 3.141592653589793238462643383279
    if z > 25:
        # Long range approximation
        F = factorial2(2*m-1)/(2**(m+1))*(pi/(z**(2*m+1)))**0.5
    elif z == 0.0:
        # special case of T = 0
        return 1.0/(2.0*m+1.0)
    else:
        F = 0.0
        temp1 = factorial2(2*m-1)
        threshold = 10**-12 # 10**-10 from purple book, but might not be in this context
        for i in range(0, 1000):
            Fcheck = F
            F += (temp1*(2*z)**i)/(factorial2(2*m+2*i+1))
            Fcheck -= F
            if abs(Fcheck) < threshold:
                break
        F *= np.exp(-z)
    return F
    
    
@jit(float64(int64,int64,int64,float64,float64,float64,float64,float64,float64),nopython=True,cache=True)
def Expansion_coefficients(i, j, t, Qx, a, b, XPA, XPB, XAB):
    #McMurchie-Davidson scheme, 9.5.6 and 9.5.7 Helgaker
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        return 0.0
    elif i == j == t == 0:
        return np.exp(-q*Qx*Qx)
    elif j == 0:
        return (1.0/(2.0*p))*E(i-1,j,t-1,Qx,a,b,XPA,XPB,XAB) + XPA*E(i-1,j,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i-1,j,t+1,Qx,a,b,XPA,XPB,XAB)
    else:
        return (1.0/(2.0*p))*E(i,j-1,t-1,Qx,a,b,XPA,XPB,XAB) + XPB*E(i,j-1,t,Qx,a,b,XPA,XPB,XAB) + (t+1.0)*E(i,j-1,t+1,Qx,a,b,XPA,XPB,XAB)
        

@jit(float64(float64[:,:]),nopython=True,cache=True)
def nuclear_nuclear_repulsion(molecule):
    #Classical nucleus nucleus repulsion
    Vnn = 0
    for i in range(1, len(molecule)):
        for j in range(1, len(molecule)):
            if i < j:
                Vnn += (molecule[i,0]*molecule[j,0])/(((molecule[i,1]-molecule[j,1])**2+(molecule[i,2]-molecule[j,2])**2+(molecule[i,3]-molecule[j,3])**2))**0.5
    return Vnn