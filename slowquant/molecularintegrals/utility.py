import numpy as np
from numba import jit, float64, int64
import math


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
        F = 1.0/(2.0*m+1.0)
    elif m == 0:
        F = (pi/(4*z))**0.5*math.erf(z)
    else:
        F = 0.0
        temp1 = factorial2(2*m-1)
        threshold = 10**-14
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
    

@jit(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:], float64[:,:,:], int64, int64, int64, int64, int64, int64),nopython=True,cache=True)
def ERI_expansion_coeff_sum(Ex1, Ey1, Ez1, Ex2, Ey2, Ez2, R, tmax, umax, vmax, taumax, numax, phimax):
    output = 0.0
    for tau in range(0, taumax):
        for nu in range(0, numax):
            for phi in range(0, phimax):
                temp = (-1.0)**(tau+nu+phi)*Ex2[tau]*Ey2[nu]*Ez2[phi]
                for t in range(0, tmax):
                    for u in range(0, umax):
                        for v in range(0, vmax):
                            output += temp*Ex1[t]*Ey1[u]*Ez1[v]*R[t+tau,u+nu,v+phi]
    return output
    

@jit(float64[:,:,:,:](int64[:],int64[:],int64[:],int64[:],float64[:,:,:,:], float64[:]),nopython=True,cache=True)
def put_in_array_ERI(idx_list_1, idx_list_2, idx_list_3, idx_list_4, ERI, new_values):
    # Not actual angular_moment
    angular_moment_1 = len(idx_list_1)
    angular_moment_2 = len(idx_list_2)
    angular_moment_3 = len(idx_list_3)
    angular_moment_4 = len(idx_list_4)
    if angular_moment_1 < angular_moment_2:
        angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        idx_list_1, idx_list_2 = idx_list_2, idx_list_1
    if angular_moment_3 < angular_moment_4:
        angular_moment_3, angular_moment_4 = angular_moment_4, angular_moment_3
        idx_list_3, idx_list_4 = idx_list_3, idx_list_4
    if angular_moment_1*(angular_moment_1+1)//2+angular_moment_2 < angular_moment_3*(angular_moment_3+1)//2+angular_moment_4:
        angular_moment_1, angular_moment_2, angular_moment_3, angular_moment_4 = angular_moment_3, angular_moment_4, angular_moment_1, angular_moment_2
        idx_list_1, idx_list_2, idx_list_3, idx_list_4 = idx_list_3, idx_list_4, idx_list_1, idx_list_2
    counter = 0
    for i in idx_list_1:
        for j in idx_list_2:
            for k in idx_list_3:
                for l in idx_list_4:
                    ERI[i,j,k,l] = ERI[j,i,k,l] = ERI[i,j,l,k] = ERI[j,i,l,k] = ERI[k,l,i,j] = ERI[k,l,j,i] = ERI[l,k,i,j] = ERI[l,k,j,i] = new_values[counter]
                    counter += 1
    return ERI
    
    
@jit(int64[:,:](int64[:],int64[:],int64[:],int64[:],int64[:,:]),nopython=True,cache=True)
def make_idx_list_two_electron(idx_list_1, idx_list_2, idx_list_3, idx_list_4, idx_array):
    # Not actual angular_moment
    angular_moment_1 = len(idx_list_1)
    angular_moment_2 = len(idx_list_2)
    angular_moment_3 = len(idx_list_3)
    angular_moment_4 = len(idx_list_4)
    if angular_moment_1 < angular_moment_2:
        angular_moment_1, angular_moment_2 = angular_moment_2, angular_moment_1
        idx_list_1, idx_list_2 = idx_list_2, idx_list_1
    if angular_moment_3 < angular_moment_4:
        angular_moment_3, angular_moment_4 = angular_moment_4, angular_moment_3
        idx_list_3, idx_list_4 = idx_list_3, idx_list_4
    if angular_moment_1*(angular_moment_1+1)//2+angular_moment_2 < angular_moment_3*(angular_moment_3+1)//2+angular_moment_4:
        angular_moment_1, angular_moment_2, angular_moment_3, angular_moment_4 = angular_moment_3, angular_moment_4, angular_moment_1, angular_moment_2
        idx_list_1, idx_list_2, idx_list_3, idx_list_4 = idx_list_3, idx_list_4, idx_list_1, idx_list_2
    counter = 0
    for i in idx_list_1:
        for j in idx_list_2:
            for k in idx_list_3:
                for l in idx_list_4:
                    idx_array[counter,0] = i
                    idx_array[counter,1] = j
                    idx_array[counter,2] = k
                    idx_array[counter,3] = l
                    counter += 1
    return idx_array[:counter]
