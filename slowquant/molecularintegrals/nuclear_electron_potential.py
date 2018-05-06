import numpy as np
from numba import jit, float64
from slowquant.molecularintegrals.utility import Normalization, boys_function


@jit(float64[:](float64[:], float64[:], float64, float64, float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_nuclear_electron_potential_0_0(Coord_1, Coord_2, guass_exp_1, gauss_exp_2, atoms, E, R, primitive):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    primitive[:] = 0.0
    
    E[0,0,0,:] = np.exp(-q*XAB*XAB)

    for i in range(0, len(atoms)):
        XPC, YPC, ZPC = P - atoms[i,1:4]
        RPC = (XPC**2 + YPC**2 + ZPC**2)**0.5
        Charge = atoms[i,0]
        R[0,0,0,0] = (-2.0*p)**0 * boys_function(0,p*RPC*RPC)

        for t in range(0, 1):
            for u in range(0, 1):
                for v in range(0, 1):
                    primitive[0] += Charge*E[0,0,t,0]*E[0,0,u,1]*E[0,0,v,2]*R[t,u,v,0]

    return -2.0*pi/p*primitive


@jit(float64[:](float64[:], float64[:], float64, float64, float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_nuclear_electron_potential_1_0(Coord_1, Coord_2, guass_exp_1, gauss_exp_2, atoms, E, R, primitive):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    primitive[:] = 0.0
    
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[0,1,1,:] = (1.0/(2.0*p)) * E[0,0,0,:]
    E[0,1,0,:] = XPB * E[0,0,0,:]
    E[1,0,1,:] = (1.0/(2.0*p)) * E[0,0,0,:]
    E[1,0,0,:] = XPA * E[0,0,0,:]

    for i in range(0, len(atoms)):
        XPC, YPC, ZPC = P - atoms[i,1:4]
        RPC = (XPC**2 + YPC**2 + ZPC**2)**0.5
        Charge = atoms[i,0]
        R[0,0,0,0] = (-2.0*p)**0 * boys_function(0,p*RPC*RPC)
        R[0,0,0,1] = (-2.0*p)**1 * boys_function(1,p*RPC*RPC)
        R[0,0,1,0] = ZPC * R[0,0,0,1]
        R[0,1,0,0] = YPC * R[0,0,0,1]
        R[1,0,0,0] = XPC * R[0,0,0,1]

        for t in range(0, 2):
            for u in range(0, 1):
                for v in range(0, 1):
                    primitive[0] += Charge*E[1,0,t,0]*E[0,0,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 2):
                for v in range(0, 1):
                    primitive[1] += Charge*E[0,0,t,0]*E[1,0,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 1):
                for v in range(0, 2):
                    primitive[2] += Charge*E[0,0,t,0]*E[0,0,u,1]*E[1,0,v,2]*R[t,u,v,0]

    return -2.0*pi/p*primitive


@jit(float64[:](float64[:], float64[:], float64, float64, float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def primitive_nuclear_electron_potential_1_1(Coord_1, Coord_2, guass_exp_1, gauss_exp_2, atoms, E, R, primitive):
    pi = 3.141592653589793238462643383279
    p = guass_exp_1 + gauss_exp_2
    q = guass_exp_1 * gauss_exp_2 / p
    P = (guass_exp_1*Coord_1 + gauss_exp_2*Coord_2) / p
    XAB = Coord_1 - Coord_2
    XPA = P - Coord_1
    XPB = P - Coord_2
    primitive[:] = 0.0
    
    E[0,0,0,:] = np.exp(-q*XAB*XAB)
    E[0,1,1,:] = (1.0/(2.0*p)) * E[0,0,0,:]
    E[0,1,0,:] = XPB * E[0,0,0,:]
    E[0,2,2,:] = (1.0/(2.0*p)) * E[0,1,1,:]
    E[0,2,1,:] = (1.0/(2.0*p)) * E[0,1,0,:] + XPB * E[0,1,1,:]
    E[0,2,0,:] = XPB * E[0,1,0,:] + 1.0 * E[0,1,1,:]
    E[1,0,1,:] = (1.0/(2.0*p)) * E[0,0,0,:]
    E[1,0,0,:] = XPA * E[0,0,0,:]
    E[1,1,2,:] = (1.0/(2.0*p)) * E[1,0,1,:]
    E[1,1,1,:] = (1.0/(2.0*p)) * E[1,0,0,:] + XPB * E[1,0,1,:]
    E[1,1,0,:] = XPB * E[1,0,0,:] + 1.0 * E[1,0,1,:]
    E[2,0,2,:] = (1.0/(2.0*p)) * E[1,0,1,:]
    E[2,0,1,:] = (1.0/(2.0*p)) * E[1,0,0,:] + XPA * E[1,0,1,:]
    E[2,0,0,:] = XPA * E[1,0,0,:] + 1.0 * E[1,0,1,:]

    for i in range(0, len(atoms)):
        XPC, YPC, ZPC = P - atoms[i,1:4]
        RPC = (XPC**2 + YPC**2 + ZPC**2)**0.5
        Charge = atoms[i,0]
        R[0,0,0,0] = (-2.0*p)**0 * boys_function(0,p*RPC*RPC)
        R[0,0,0,1] = (-2.0*p)**1 * boys_function(1,p*RPC*RPC)
        R[0,0,1,0] = ZPC * R[0,0,0,1]
        R[0,0,0,2] = (-2.0*p)**2 * boys_function(2,p*RPC*RPC)
        R[0,0,1,1] = ZPC * R[0,0,0,2]
        R[0,0,2,0] = 1 * R[0,0,0,1] + ZPC * R[0,0,1,1]
        R[0,1,0,0] = YPC * R[0,0,0,1]
        R[0,1,1,0] = YPC * R[0,0,1,1]
        R[0,1,0,1] = YPC * R[0,0,0,2]
        R[0,2,0,0] = 1 * R[0,0,0,1] + YPC * R[0,1,0,1]
        R[1,0,0,0] = XPC * R[0,0,0,1]
        R[1,0,1,0] = XPC * R[0,0,1,1]
        R[1,1,0,0] = XPC * R[0,1,0,1]
        R[1,0,0,1] = XPC * R[0,0,0,2]
        R[2,0,0,0] = 1 * R[0,0,0,1] + XPC * R[1,0,0,1]

        for t in range(0, 3):
            for u in range(0, 1):
                for v in range(0, 1):
                    primitive[0] += Charge*E[1,1,t,0]*E[0,0,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 2):
            for u in range(0, 2):
                for v in range(0, 1):
                    primitive[1] += Charge*E[1,0,t,0]*E[0,1,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 2):
            for u in range(0, 1):
                for v in range(0, 2):
                    primitive[2] += Charge*E[1,0,t,0]*E[0,0,u,1]*E[0,1,v,2]*R[t,u,v,0]
        for t in range(0, 2):
            for u in range(0, 2):
                for v in range(0, 1):
                    primitive[3] += Charge*E[0,1,t,0]*E[1,0,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 3):
                for v in range(0, 1):
                    primitive[4] += Charge*E[0,0,t,0]*E[1,1,u,1]*E[0,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 2):
                for v in range(0, 2):
                    primitive[5] += Charge*E[0,0,t,0]*E[1,0,u,1]*E[0,1,v,2]*R[t,u,v,0]
        for t in range(0, 2):
            for u in range(0, 1):
                for v in range(0, 2):
                    primitive[6] += Charge*E[0,1,t,0]*E[0,0,u,1]*E[1,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 2):
                for v in range(0, 2):
                    primitive[7] += Charge*E[0,0,t,0]*E[0,1,u,1]*E[1,0,v,2]*R[t,u,v,0]
        for t in range(0, 1):
            for u in range(0, 1):
                for v in range(0, 3):
                    primitive[8] += Charge*E[0,0,t,0]*E[0,0,u,1]*E[1,1,v,2]*R[t,u,v,0]

    return -2.0*pi/p*primitive


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:], float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def nuclear_electron_integral_0_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, atoms, E_buffer, R_buffer, primitives_buffer_2):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:1]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_0_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:], float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def nuclear_electron_integral_1_0(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, atoms, E_buffer, R_buffer, primitives_buffer_2):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:3]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_1_0(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(0.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])
    output_buffer[1] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])
    output_buffer[2] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:,:,:], float64[:], float64[:], float64[:,:], float64[:,:,:,:], float64[:,:,:,:], float64[:]), nopython=True, cache=True)
def nuclear_electron_integral_1_1(Coord_1, Coord_2, gauss_exp_1, gauss_exp_2, Contra_coeffs_1, Contra_coeffs_2, output_buffer, primitives_buffer, Contraction_1_buffer, Contraction_2_buffer, atoms, E_buffer, R_buffer, primitives_buffer_2):
    number_primitive_1 = len(gauss_exp_1)
    number_primitive_2 = len(gauss_exp_2)
    Contraction_1_buffer = Contraction_1_buffer[:number_primitive_1]
    Contraction_2_buffer = Contraction_2_buffer[:number_primitive_2]
    primitives_buffer = primitives_buffer[:number_primitive_1,:number_primitive_2,:9]
    for i in range(0, len(gauss_exp_1)):
        for j in range(0, len(gauss_exp_2)):
            primitives_buffer[i,j,:] = primitive_nuclear_electron_potential_1_1(Coord_1, Coord_2, gauss_exp_1[i], gauss_exp_2[j], atoms, E_buffer, R_buffer, primitives_buffer_2)
    for i in range(0, len(Contra_coeffs_1)):
        Contraction_1_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_1[i]) * Contra_coeffs_1[i]
    for i in range(0, len(Contra_coeffs_2)):
        Contraction_2_buffer[i] = Normalization(1.0,0.0,0.0,gauss_exp_2[i]) * Contra_coeffs_2[i]
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,0])
    output_buffer[0] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,1])
    output_buffer[1] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,2])
    output_buffer[2] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,3])
    output_buffer[3] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,4])
    output_buffer[4] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,5])
    output_buffer[5] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,6])
    output_buffer[6] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,7])
    output_buffer[7] = np.dot(temp, Contraction_2_buffer)
    temp = np.dot(Contraction_1_buffer, primitives_buffer[:,:,8])
    output_buffer[8] = np.dot(temp, Contraction_2_buffer)
    return output_buffer


