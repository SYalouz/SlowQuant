import numpy as np
from numba import jit, float64


@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)
def E_0_0_0(q, p12, XAB, XPA, XPB, E):
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    return E


@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)
def E_1_0_0(q, p12, XAB, XPA, XPB, E):
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[1,0,0,:] = XPA * E[0,0,0,:]
    return E


@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)
def E_1_0_1(q, p12, XAB, XPA, XPB, E):
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[1,0,1,:] = p12 * E[0,0,0,:]
    E[1,0,0,:] = XPA * E[0,0,0,:]
    return E


@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)
def E_1_1_0(q, p12, XAB, XPA, XPB, E):
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[0,1,0,:] = XPB * E[0,0,0,:]
    E[1,0,0,:] = XPA * E[0,0,0,:]
    E[1,0,1,:] = p12 * E[0,0,0,:]
    E[1,1,0,:] = XPB * E[1,0,0,:] + E[1,0,1,:]
    return E


@jit(float64[:,:,:,:](float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]), nopython=True, cache=True)
def E_1_1_2(q, p12, XAB, XPA, XPB, E):
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[0,1,1,:] = p12 * E[0,0,0,:]
    E[0,1,0,:] = XPB * E[0,0,0,:]
    E[1,0,1,:] = p12 * E[0,0,0,:]
    E[1,0,0,:] = XPA * E[0,0,0,:]
    E[1,1,2,:] = p12 * E[1,0,1,:]
    E[1,1,1,:] = p12 * E[1,0,0,:] + XPB * E[1,0,1,:]
    E[1,1,0,:] = XPB * E[1,0,0,:] + E[1,0,1,:]
    return E


