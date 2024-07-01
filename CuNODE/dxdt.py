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


class dxdt:
    def __init__(self):
        @cuda.jit(float64(float64,
                          float64),
                  device=True,
                  inline=True)
        def clip(value,
                 clip_value):
            if value <= clip_value and value >= -clip_value:
                return value
            elif value > clip_value:
                return clip_value
            else:
                return -clip_value
        
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
            outarray[0] = state[1]
            outarray[1] = (-state[0] - constants[3]*state[1] + constants[0] * state[2] + constants[7] * ref)
            outarray[2] = (-constants[1] * state[ 2] + constants[2] * state[3] * state[3])
            outarray[3] = (-constants[6] * state[3] + constants[6] * control)
            outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])
        
        
        @cuda.jit((float64[:],
                    float64[:],
                    xoro_type[:]
                    ),
                  device=True,
                  inline=True)
        def get_noise(noise_array,
                      sigmas,
                      RNG):
        
            for i in range(len(noise_array)):
                if sigmas[i] != 0.0:
                    noise_array[i] = xoroshiro128p_normal_float64(RNG, i) * sigmas[i]
                else:
                    noise_array[i] = float64(0.0)
                    
        self.dxdtfunc = dxdtfunc
        self.getnoisefunc = get_noise
        self.clipfunc = clip



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