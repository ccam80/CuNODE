# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:50:05 2024

@author: cca79
"""
import numpy as np

""" This file contains the system parameters for each cantilever. It's hard work,
as I couldn't find a way for an object or something tidy to hold all of this info
in a way that was easily modified, without breaking CUDA. It's my fault, not CUDAs."""


#Simulation parameters
step_size = 0.001
fs = 10
duration = np.float64(10)


# System Params
alpha = np.float64(0.52)
beta = np.float64(0.0133)
gamma = np.float64(0.0624)
delta = np.float64(0.012)
omega = np.float64(1.0)
omH = np.float64(1/114)
omL = np.float64(2.0)
pz_sign = np.float64(1)
hpf_sign = np.float64(1)
rhat = np.float64(0.01)
cliplevel = np.float64(1)
a = np.float64(1)
b = np.float64(-0.4)
delta_omega = delta*omega
omega_sq = omega**2

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
                           10/8191,
                           0.0,
                           0.0,
                           0.0],
                          dtype=np.float64)




#init values
init_disp = np.float64(1.0)
init_vel = np.float64(0.0)
init_temp = np.float64(1.0)
init_ctrl = np.float64(0.0)
init_hpf = np.float64(1.0)

#inits array for passing to naiive version
inits = np.asarray([init_disp,
                    init_vel,
                    init_temp,
                    init_ctrl,
                    init_hpf])

#Indices for each cell during update, as worked out in matriculation.xlsx, interpreted
# and typeset by ChatGPT, and  hated by everyone.
update_indices_constant = np.asarray([[[ 0,  0],
                                         [ 3,  7],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 1, 11],
                                         [ 0,  7],
                                         [ 0,  0],
                                         [ 0,  7],
                                         [ 0,  0],
                                         [ 0, 11],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 1, 11],
                                         [ 1,  3],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 1, 11],
                                         [ 4,  3],
                                         [ 0,  0],
                                         [ 1,  7],
                                         [ 0,  0],
                                         [ 5,  7],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 5, 11],
                                         [ 2,  7],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 2,  7],
                                         [ 2,  3],
                                         [ 0,  0],
                                         [ 2,  7],
                                         [ 0,  0],
                                         [ 2, 11],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 6, 11],
                                         [ 3,  7],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  3],
                                         [ 3,  3],
                                         [ 0,  0],
                                         [ 3,  7],
                                         [ 0,  0],
                                         [ 3, 11],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 1, 11],
                                         [ 6,  3],
                                         [ 0,  0],
                                         [ 4,  7],
                                         [ 0,  0],
                                         [ 4, 11],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 0,  7],
                                         [ 1, 11],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 2,  7],
                                         [ 5,  3],
                                         [ 0,  0],
                                         [ 3,  7],
                                         [ 3,  7],
                                         [ 0,  0],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 4,  7],
                                         [ 4,  7],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 4,  7],
                                         [ 0,  0],
                                         [ 0,  0]],

                                        [[ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0],
                                         [ 0,  0]]],
                                        dtype=np.int16)

#Constants arranged in lookuptable form for the overcomplicated version
working_array_constants = np.asarray([
                                    [-omL*step_size, 0, 0, 0,
                                     step_size, 0, 0, init_disp,
                                     0, 1.0, 0, 0],
                                    [ 0, 0, 0, 0,
                                     -delta*omega*step_size, 0, 0, 0,
                                     0, 0, 0, init_vel],
                                    [gamma*step_size, 0, 0, 0,
                                     -beta*step_size, 0, 0, init_temp,
                                     0, 0, 0, 0],
                                    [omL*step_size, 0, 0, 0,
                                     1.0, 0, 0, init_ctrl,
                                     0, 0, 0, 0],
                                    [rhat*pz_sign*step_size, 0, 0, 0,
                                     step_size * hpf_sign, 0, 0, init_hpf,
                                     0, 0, 0, 0],
                                    [-step_size*omega**2, 0, 0, 0,
                                     alpha*step_size, 0, 0, 0,
                                     0, 0, 0, 0],
                                    [-omH*step_size, 0, 0, 0,
                                     0, 0, 0, 0,
                                     a, 0, b, 0],
                                    [0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0]],
                                    dtype=np.float64)