import numpy as np
<<<<<<< HEAD
from numba import jit, float64
from slowquant.molecularintegrals.utility import boys_function, boys_function_n_zero


@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
from slowquant.molecularintegrals.utility import boys_function, boys_function_n_zero


>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_0_0_0_0(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    return R[:,:,:,0]


<<<<<<< HEAD
@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_1_0_0_0(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    R_0_0_0_1 = -2.0*p * boys_function(1,p*RPC*RPC)
    R[0,0,1] = ZPC * R_0_0_0_1
    R[0,1,0] = YPC * R_0_0_0_1
    R[1,0,0] = XPC * R_0_0_0_1
    return R[:,:,:,0]


<<<<<<< HEAD
@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_1_0_1_0(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    R_0_0_0_1 = -2.0*p * boys_function(1,p*RPC*RPC)
    R[0,0,1] = ZPC * R_0_0_0_1
    R_0_0_0_2 = (-2.0*p)**2.0 * boys_function(2,p*RPC*RPC)
    R_0_0_1_1 = ZPC * R_0_0_0_2
    R[0,0,2] = R_0_0_0_1 + ZPC * R_0_0_1_1
    R[0,1,0] = YPC * R_0_0_0_1
    R[0,1,1] = YPC * R_0_0_1_1
    R_0_1_0_1 = YPC * R_0_0_0_2
    R[0,2,0] = R_0_0_0_1 + YPC * R_0_1_0_1
    R[1,0,0] = XPC * R_0_0_0_1
    R[1,0,1] = XPC * R_0_0_1_1
    R[1,1,0] = XPC * R_0_1_0_1
    R[2,0,0] = R_0_0_0_1 + XPC * (XPC * R_0_0_0_2)
    return R[:,:,:,0]


<<<<<<< HEAD
@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_1_1_0_0(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    R_0_0_0_1 = -2.0*p * boys_function(1,p*RPC*RPC)
    R[0,0,1] = ZPC * R_0_0_0_1
    R_0_0_0_2 = (-2.0*p)**2.0 * boys_function(2,p*RPC*RPC)
    R_0_0_1_1 = ZPC * R_0_0_0_2
    R[0,0,2] = R_0_0_0_1 + ZPC * R_0_0_1_1
    R[0,1,0] = YPC * R_0_0_0_1
    R[0,1,1] = YPC * R_0_0_1_1
    R_0_1_0_1 = YPC * R_0_0_0_2
    R[0,2,0] = R_0_0_0_1 + YPC * R_0_1_0_1
    R[1,0,0] = XPC * R_0_0_0_1
    R[1,0,1] = XPC * R_0_0_1_1
    R[1,1,0] = XPC * R_0_1_0_1
    R[2,0,0] = R_0_0_0_1 + XPC * (XPC * R_0_0_0_2)
    return R[:,:,:,0]


<<<<<<< HEAD
@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_1_1_1_0(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    R_0_0_0_1 = -2.0*p * boys_function(1,p*RPC*RPC)
    R[0,0,1] = ZPC * R_0_0_0_1
    R_0_0_0_2 = (-2.0*p)**2.0 * boys_function(2,p*RPC*RPC)
    R_0_0_1_1 = ZPC * R_0_0_0_2
    R[0,0,2] = R_0_0_0_1 + ZPC * R_0_0_1_1
    R_0_0_0_3 = (-2.0*p)**3.0 * boys_function(3,p*RPC*RPC)
    R_0_0_1_2 = ZPC * R_0_0_0_3
    R_0_0_2_1 = R_0_0_0_2 + ZPC * R_0_0_1_2
    R[0,0,3] = 2.0 * R_0_0_1_1 + ZPC * R_0_0_2_1
    R[0,1,0] = YPC * R_0_0_0_1
    R[0,1,1] = YPC * R_0_0_1_1
    R[0,1,2] = YPC * R_0_0_2_1
    R_0_1_0_1 = YPC * R_0_0_0_2
    R[0,2,0] = R_0_0_0_1 + YPC * R_0_1_0_1
    R_0_1_1_1 = YPC * R_0_0_1_2
    R[0,2,1] = R_0_0_1_1 + YPC * R_0_1_1_1
    R_0_1_0_2 = YPC * R_0_0_0_3
    R_0_2_0_1 = R_0_0_0_2 + YPC * R_0_1_0_2
    R[0,3,0] = 2.0 * R_0_1_0_1 + YPC * R_0_2_0_1
    R[1,0,0] = XPC * R_0_0_0_1
    R[1,0,1] = XPC * R_0_0_1_1
    R[1,0,2] = XPC * R_0_0_2_1
    R[1,1,0] = XPC * R_0_1_0_1
    R[1,1,1] = XPC * R_0_1_1_1
    R[1,2,0] = XPC * R_0_2_0_1
    R_1_0_0_1 = XPC * R_0_0_0_2
    R[2,0,0] = R_0_0_0_1 + XPC * R_1_0_0_1
    R[2,0,1] = R_0_0_1_1 + XPC * (XPC * R_0_0_1_2)
    R[2,1,0] = R_0_1_0_1 + XPC * (XPC * R_0_1_0_2)
    R[3,0,0] = 2.0 * R_1_0_0_1 + XPC * (R_0_0_0_2 + XPC * (XPC * R_0_0_0_3))
    return R[:,:,:,0]


<<<<<<< HEAD
@jit(float64[:,:,:](float64, float64, float64, float64, float64, float64[:,:,:,:]), nopython=True, cache=True)
=======
>>>>>>> f4aef438580fb18e556a59f1ef22e55c6bd341e9
def R_1_1_1_1(p, XPC, YPC, ZPC, RPC, R):
    R[0,0,0] = boys_function_n_zero(p*RPC*RPC)
    R_0_0_0_1 = -2.0*p * boys_function(1,p*RPC*RPC)
    R[0,0,1] = ZPC * R_0_0_0_1
    R_0_0_0_2 = (-2.0*p)**2.0 * boys_function(2,p*RPC*RPC)
    R_0_0_1_1 = ZPC * R_0_0_0_2
    R[0,0,2] = R_0_0_0_1 + ZPC * R_0_0_1_1
    R_0_0_0_3 = (-2.0*p)**3.0 * boys_function(3,p*RPC*RPC)
    R_0_0_1_2 = ZPC * R_0_0_0_3
    R_0_0_2_1 = R_0_0_0_2 + ZPC * R_0_0_1_2
    R[0,0,3] = 2.0 * R_0_0_1_1 + ZPC * R_0_0_2_1
    R_0_0_0_4 = (-2.0*p)**4.0 * boys_function(4,p*RPC*RPC)
    R_0_0_1_3 = ZPC * R_0_0_0_4
    R_0_0_2_2 = R_0_0_0_3 + ZPC * R_0_0_1_3
    R_0_0_3_1 = 2.0 * R_0_0_1_2 + ZPC * R_0_0_2_2
    R[0,0,4] = 3.0 * R_0_0_2_1 + ZPC * R_0_0_3_1
    R[0,1,0] = YPC * R_0_0_0_1
    R[0,1,1] = YPC * R_0_0_1_1
    R[0,1,2] = YPC * R_0_0_2_1
    R[0,1,3] = YPC * R_0_0_3_1
    R_0_1_0_1 = YPC * R_0_0_0_2
    R[0,2,0] = R_0_0_0_1 + YPC * R_0_1_0_1
    R_0_1_1_1 = YPC * R_0_0_1_2
    R[0,2,1] = R_0_0_1_1 + YPC * R_0_1_1_1
    R_0_1_2_1 = YPC * R_0_0_2_2
    R[0,2,2] = R_0_0_2_1 + YPC * R_0_1_2_1
    R_0_1_0_2 = YPC * R_0_0_0_3
    R_0_2_0_1 = R_0_0_0_2 + YPC * R_0_1_0_2
    R[0,3,0] = 2.0 * R_0_1_0_1 + YPC * R_0_2_0_1
    R_0_1_1_2 = YPC * R_0_0_1_3
    R_0_2_1_1 = R_0_0_1_2 + YPC * R_0_1_1_2
    R[0,3,1] = 2.0 * R_0_1_1_1 + YPC * R_0_2_1_1
    R_0_1_0_3 = YPC * R_0_0_0_4
    R_0_2_0_2 = R_0_0_0_3 + YPC * R_0_1_0_3
    R_0_3_0_1 = 2.0 * R_0_1_0_2 + YPC * R_0_2_0_2
    R[0,4,0] = 3.0 * R_0_2_0_1 + YPC * R_0_3_0_1
    R[1,0,0] = XPC * R_0_0_0_1
    R[1,0,1] = XPC * R_0_0_1_1
    R[1,0,2] = XPC * R_0_0_2_1
    R[1,0,3] = XPC * R_0_0_3_1
    R[1,1,0] = XPC * R_0_1_0_1
    R[1,1,1] = XPC * R_0_1_1_1
    R[1,1,2] = XPC * R_0_1_2_1
    R[1,2,0] = XPC * R_0_2_0_1
    R[1,2,1] = XPC * R_0_2_1_1
    R[1,3,0] = XPC * R_0_3_0_1
    R_1_0_0_1 = XPC * R_0_0_0_2
    R[2,0,0] = R_0_0_0_1 + XPC * R_1_0_0_1
    R_1_0_1_1 = XPC * R_0_0_1_2
    R[2,0,1] = R_0_0_1_1 + XPC * R_1_0_1_1
    R[2,0,2] = R_0_0_2_1 + XPC * (XPC * R_0_0_2_2)
    R_1_1_0_1 = XPC * R_0_1_0_2
    R[2,1,0] = R_0_1_0_1 + XPC * R_1_1_0_1
    R[2,1,1] = R_0_1_1_1 + XPC * (XPC * R_0_1_1_2)
    R[2,2,0] = R_0_2_0_1 + XPC * (XPC * R_0_2_0_2)
    R_1_0_0_2 = XPC * R_0_0_0_3
    R_2_0_0_1 = R_0_0_0_2 + XPC * R_1_0_0_2
    R[3,0,0] = 2.0 * R_1_0_0_1 + XPC * R_2_0_0_1
    R[3,0,1] = 2.0 * R_1_0_1_1 + XPC * (R_0_0_1_2 + XPC * (XPC * R_0_0_1_3))
    R[3,1,0] = 2.0 * R_1_1_0_1 + XPC * (R_0_1_0_2 + XPC * (XPC * R_0_1_0_3))
    R[4,0,0] = 3.0 * R_2_0_0_1 + XPC * (2.0 * R_1_0_0_2 + XPC * (R_0_0_0_3 + XPC * (XPC * R_0_0_0_4)))
    return R[:,:,:,0]


