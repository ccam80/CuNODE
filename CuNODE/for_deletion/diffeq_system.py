# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:42:35 2024

@author: cca79
"""
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"

from numba import cuda, float64, int64, int32, float32, from_dtype
from numpy import asarray
from _utils import clamp_32, clamp_64
import numpy as np
from math import cos
from numba import from_dtype

class system_constant_class(dict):
    def set_constant(self, key, item):
        if key in self:
            self[key] = item
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")

    def get_constant(self, key):
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Constant {key} not in constants dictionary")



def system_constants(constants_dict=None, **kwargs):

    constants = system_constant_class()

    defaults = {'alpha':0.52,
                'beta' : 0.0133,
                'gamma' : 0.0624,
                'delta' : 0.012,
                'omega_n' : 1.0,
                'omH' : 1/114,
                'omL' : 2.0,
                'pz_sign' : 1.0,
                'hpf_sign' : -1.0,
                'rhat' : 0.01,
                'cliplevel' : 1,
                'a' : 2,
                'b' : -0.4,
                'omega_forcing': 1.0}


    if constants_dict is None:
        constants_dict = {}

    combined_updates = {**defaults, **constants_dict, **kwargs}

    # Note: If the same value occurs in the dict and
    # keyword args, the kwargs one will win.
    for key, item in combined_updates.items():
        constants.update(combined_updates)

    return constants


class diffeq_system:
    """ This class should contain all system definitions. The constants management
    scheme can be a little tricky, because the GPU stuff can't handle dictionaries.
    The constants_array will be passed to your dxdt function - you can use the indices
    given in self.constant_indices to map them out while you set up your dxdt function.

    > test_system = diffeq_system()
    > print(diffeq_system.constant_indices)

    - Place all of your system constants and their labelsin the constants_dict.
    - Update self.num_states to match the number of state variables/ODEs you
    need to solve.
    - Feel free to define any helper functions inside the __init__ function.
    These must have the cuda.jit decorator with a signature (return(arg)), like you can
    see in the example functions.
    You can call these in the dxdt function.
    - update noise_sigmas with the std dev of gaussian noise in any state if
    you're doing a "noisy" run.

    Many numpy (and other) functions won't work inside the dxdt or CUDA device
    functions. Try using the Cupy function instead if you get an error.

    """
    def __init__(self,
                 num_states = 5,
                 precision=np.float64,
                 **kwargs):
        """Set system constant values then function as a factory function to
        build CUDA device functions for use in the ODE solver kernel. No
        arguments, no returns it's all just bad coding practice in here.

        Everything except for the constants_array and constant_indices generators
        and dxdt assignment at the end is an example, you will need to overwrite"""

        self.num_states = 5
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        self.noise_sigmas = np.zeros(self.num_states, dtype=precision)

        self.constants_dict  = system_constants(kwargs)

        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=precision)
        self.constant_indices = {label: index for index, (label, constant) in enumerate(self.constants_dict.items())}



        if self.numba_precision == float32:
            clamp = clamp_32
        else:
            clamp = clamp_64

        @cuda.jit(self.numba_precision(self.numba_precision,
                                       self.numba_precision,
                                       self.numba_precision,
                                       self.numba_precision),
                  device=True,
                  inline=True)
        def linear_control_eq(a,
                              b,
                              feedback_state,
                              cliplevel):
            return clamp(a * feedback_state + b, cliplevel)


        @cuda.jit((self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision[:],
                   self.numba_precision),
                  device=True,
                  inline=True,)
        def dxdtfunc(outarray,
                     state,
                     constants,
                     t):
            """ Put your dxdt calculations in here, including any reference signal
            or other math. Ugly is good here, avoid creating local variables and
            partial calculations - a long string of multiplies and adds, referring to
            the same array, might help the compiler make it fast. Avoid low powers,
            use consecutive multiplications instead.

            For a list of supported math functions you can include, see
            :https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html"""

            #reference = rhat * sin(wt), clipped to replicate a hardware limitation
            ref = clamp(cuda.libdevice.cos(precision(constants[13]*t)) * constants[9], constants[10])

            #change control function to try different controllers
            control = linear_control_eq(constants[11], constants[12], state[4], constants[10])

            outarray[0] = state[1]
            outarray[1] = (-state[0] - constants[3]*state[1] + constants[0] * state[2] + constants[7] * ref)
            outarray[2] = (-constants[1] * state[ 2] + constants[2] * state[3] * state[3])
            outarray[3] = (-constants[6] * state[3] + constants[6] * control)
            outarray[4] = (-constants[5] * state[4] + constants[8] * state[1])


        self.dxdtfunc = dxdtfunc
        self.clipfunc = clamp


    def update_constants(self, updates_dict=None, **kwargs):
        if updates_dict is None:
            updates_dict = {}

        combined_updates = {**updates_dict, **kwargs}

        # Note: If the same value occurs in the dict and
        # keyword args, the kwargs one will win.
        for key, item in combined_updates.items():
            self.constants_dict.set_constant(key, self.precision(item))

        self.constants_array = asarray([constant for (label, constant) in self.constants_dict.items()], dtype=self.precision)

    def set_noise_sigmas(self, noise_vector):
        self.noise_sigmas = np.asarray(noise_vector, dtype=self.precision)

    def get_noise_sigmas(self):
        return self.noise_sigmas.copy()
#******************************* TEST CODE ******************************** #
# if __name__ == '__main__':
    # sys = diffeq_system()
    # dxdt = sys.dxdtfunc

    # @cuda.jit()
    # def testkernel(out):
    #     # precision = np.float32
    #     # numba_precision = float32
    #     l_dxdt = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
    #     l_states = cuda.local.array(shape=NUM_STATES, dtype=numba_precision)
    #     l_constants = cuda.local.array(shape=NUM_CONSTANTS, dtype=numba_precision)
    #     l_states[:] = precision(1.0)
    #     l_constants[:] = precision(1.0)

    #     t = precision(1.0)
    #     dxdt(l_dxdt,
    #         l_states,
    #         l_constants,
    #         t)

    #     out = l_dxdt


    #     NUM_STATES = 5
    #     NUM_CONSTANTS = 14
    #     outtest = np.zeros(NUM_STATES, dtype=np.float4)
    #     out = cuda.to_device(outtest)
    #     print("Testing to see if your dxdt function compiles using CUDA...")
    #     testkernel[1,1](out)
    #     cuda.synchronize()
    #     out.copy_to_host(outtest)
    #     print(outtest)
