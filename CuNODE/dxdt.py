# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:42:35 2024

@author: cca79
"""
from numba import cuda, float64, int64, int32, float32, from_dtype
from numba.cuda.random import xoroshiro128p_normal_float64, xoroshiro128p_dtype
from numba import types
import numpy as np
xoro_type = from_dtype(xoroshiro128p_dtype)


#Simulation parameters
step_size = 0.001
fs = 10
duration = np.float64(1000)


# System Params
alpha = np.float64(0.52)
beta = np.float64(0.0133)
gamma = np.float64(0.0624)
delta = np.float64(0.012)
omega = np.float64(1.0)
omH = np.float64(1/114)
omL = np.float64(2.0)
pz_sign = np.float64(1)
hpf_sign = np.float64(-1)
rhat = np.float64(0.01)
cliplevel = np.float64(1)
a = np.float64(1)
b = np.float64(-0.4)


#System params combined into an array for easier import into the naiive version
constants = np.asarray([alpha,    #0
                        beta,     #1
                        gamma,    #2
                        delta,    #3
                        omega,    #4
                        omH,      #5
                        omL,      #6
                        pz_sign,  #7
                        hpf_sign, #8
                        rhat,     #9
                        cliplevel,#10
                        a,        #11
                        b,        #12
                        step_size], #13
                       dtype=np.float64)

#Std dev of noise per-state (states 5-7 are padding)
noise_sigmas = np.asarray([0.0,
                           0.0,
                           0.0,
                           0.0,
                           0/8191,
                           0.0,
                           0.0,
                           0.0],
                          dtype=np.float64)

class diffeq_system:
    def __init__(self):
        self.constants_dict  = {
            'alpha':0.52,
            'beta' : 0.0133,
            'gamma' : 0.0624,
            'delta' : 0.012,
            'omega' : 1.0,
            'omH' : 1/114,
            'omL' : 2.0,
            'pz_sign' : 1.0,
            'hpf_sign' : -1.0,
            'rhat' : 0.01,
            'cliplevel' : 1,
            'a' : 1,
            'b' : -0.4
            }
        self.constants_list = [constant for constant in self.constants_dict.items()] 
        #This should be in a utils module somewhere I think, it's not part of the system
        @cuda.jit(float64(float64,
                          float64),
                  device=True,
                  inline=True)
        def clamp(value,
                 clip_value):
            if value <= clip_value and value >= -clip_value:
                return value
            elif value > clip_value:
                return clip_value
            else:
                return -clip_value
        
        
        @cuda.jit(float64(float64,
                          float64,
                          float64),
                  device=True,
                  inline=True)
        def linear_control_eq(a,
                              b,
                              feedback_state):
            return clamp(a * feedback_state + b, constants[10])
        
        
        @cuda.jit((float64[:],
                    float64[:],
                    int32,
                    xoro_type[:]
                    ),
                  device=True,
                  inline=True)
        def get_noise(noise_array,
                      sigmas,
                      idx,
                      RNG):
        
            for i in range(len(noise_array)):
                if sigmas[i] != 0.0:
                    noise_array[i] = xoroshiro128p_normal_float64(RNG, idx) * sigmas[i]
                
        @cuda.jit((float64[:],
                   float64[:],
                   float64[:],
                   float64,
                   float64),
                  device=True,
                  inline=True,)
        def dxdtfunc(outarray,
                 state,
                 constants,
                 control,
                 ref):
            
            control = linear_control_eq(constants[10], constants[11], state[4])
            
            outarray[0] = state[1]
            outarray[1] = (-state[0] - constants[3]*state[1] + constants[0] * state[2] + constants[7] * ref)
            outarray[2] = (-constants[1] * state[ 2] + constants[2] * state[3] * state[3])
            outarray[3] = (-constants[6] * state[3] + constants[6] * control)
            outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])
            
            
        self.dxdtfunc = dxdtfunc
        self.getnoisefunc = get_noise
        self.clipfunc = clamp



# ******************************* TEST CODE ******************************** #
# @cuda.jit()
# def testkernel(out):
#     l_dxdt = cuda.local.array(shape=NUM_STATES, dtype=types.float64)
#     l_states = cuda.local.array(shape=NUM_STATES, dtype=types.float64)
#     l_constants = cuda.local.array(shape=NUM_CONSTANTS, dtype=types.float64)
#     l_states[:] = 1.0
#     l_constants[:] = 1.0

#     control=1.0
#     ref=1.0

#     dxdt(l_dxdt,
#          l_states,
#          l_constants,
#          control,
#          ref)

#     out = l_dxdt

# if __name__ == "__main__":
#     NUM_STATES = 5
#     NUM_CONSTANTS = 9
#     outtest = np.zeros(NUM_STATES, dtype=np.float64)
#     out = cuda.to_device(outtest)
#     print("Testing to see if your dxdt function compiles using CUDA...")
#     testkernel[128,1](out)
#     cuda.synchronize()
#     out.copy_to_host(outtest)
#     print(outtest)