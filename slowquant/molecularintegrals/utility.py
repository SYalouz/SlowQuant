import numpy as np
from numba import jit, float64, int64


@jit(float64(float64),nopython=True,cache=True)
def factorial2(n):
    n_range = int(n)
    out = 1.0
    if n > 0:
        for i in range(0, int(n_range+1)//2):
            out = out*(n-2*i)
    return out
    

@jit(float64(float64, float64, float64, float64),nopython=True,cache=True)
def Normalization(l, m, n, c):
    """
    Calculates the normalizations coefficients of the basisfunctions.
    """
    pi = 3.141592653589793238462643383279
    # Normalize primitive functions
    part1 = (2.0/pi)**(3.0/4.0)
    part2 = 2.0**(l+m+n) * c**((2.0*l+2.0*m+2.0*n+3.0)/(4.0))
    part3 = (factorial2(int(2*l-1))*factorial2(int(2*m-1))*factorial2(int(2*n-1)))**0.5
    N = part1 * ((part2)/(part3))
    return N


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
        threshold = 10**-12
        for i in range(0, 1000):
            Fcheck = F
            F += (temp1*(2*z)**i)/(factorial2(2*m+2*i+1))
            Fcheck -= F
            if abs(Fcheck) < threshold:
                break
        F *= np.exp(-z)
    return F
    

@jit(float64(float64[:,:]),nopython=True,cache=True)
def nuclear_nuclear_repulsion(molecule):
    #Classical nucleus nucleus repulsion
    Vnn = 0
    for i in range(1, len(molecule)):
        for j in range(1, len(molecule)):
            if i < j:
                Vnn += (molecule[i,0]*molecule[j,0])/(((molecule[i,1]-molecule[j,1])**2+(molecule[i,2]-molecule[j,2])**2+(molecule[i,3]-molecule[j,3])**2))**0.5
    return Vnn

    
@jit(nopython=True,cache=True)
def transform_to_spherical():
    None