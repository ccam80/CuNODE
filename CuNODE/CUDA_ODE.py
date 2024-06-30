# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""
from numba import cuda, from_dtype
from numba import float32, float64, int32, int64
from dxdt import dxdt
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
xoro_type = from_dtype(xoroshiro128p_dtype)



class CUDA_ODE(object):

    def __init__(self,
                 nstates,
                 dxdt=dxdt,
                 dtype = np.float64
                 ):
        self.nstates = nstates
        self.dxdt = dxdt

        self.noise_array = cuda.to_device(np.zeros(noise_array, dtype=dtype))
        self.dxdt_array


    @staticmethod()
    def dxdt():
        dxdt()

    def.noise(self):
        sigmas = self.noise_sigmas
        noise_array
    @staticmethod()
    @cuda.jit((float64[:],
                float64[:],
                xoro_type[:],
                int32),
              device=True,
              inline=True)
    def get_noise(noise_array,
                  sigmas,
                  RNG,
                  firstidx):

        for i in range(len(noise_array)):
            if sigmas[i] != 0.0:
                noise_array[i] = xoroshiro128p_normal_float64(RNG, i) * sigmas[i]
            else:
                noise_array[i] = float64(0.0)

    @staticmethod
    @cuda.jit(float64[:](float64[:],
                         float64[:],
                         float64[:],
                         xoro_type[:],
                         int32,
                         float64,
                         float64),
              device=True,
              inline=True)
    def eulermaruyama(state,
                      dxdt,
                      stochastic_sigmas,
                      stochastic_gen,
                      threadidx,
                      ref,
                      control):
        return state