# -*- coding: utf-8 -*-
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"


from numba import cuda, float64, int64, void, int32, int16, uint64, from_dtype
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
import numpy as np
from filter_coefficients import ds_100_filter
from time import time
from system_parallelisation import constants, inits, fs, duration, noise_sigmas

#This is getting a signature of the RNG datatype for the compiler, disregard if you're not using randomness.
xoro_type = from_dtype(xoroshiro128p_dtype)



BLOCKSIZE_X = 8 * 32        #Number of param sets per block
NUM_STATES = 5

# #Setting up grid of params to simulate with
# a_gains = np.asarray([i * 0.2 for i in range(-1, 1)], dtype=np.float64)
# b_params = np.asarray([i * 0.1 for i in range(-1, 1)], dtype=np.float64)
# grid_params = [(a, b) for a in a_gains for b in b_params]
grid_params = [(1,0.5), (1,0)]

#Bring in filter coefficients for polyphase decimator - FIR filter that downsamples
#TODO: Consider scaling.
filtercoefficients = np.asarray(ds_100_filter, dtype=np.float64)

# This was brought in in an array to save clutter, fetch step size for size calcs
step_size = constants[-1]

NUMTHREADS = len(grid_params)
BLOCKSPERGRID = int(max(1, np.ceil(len(grid_params) / BLOCKSIZE_X)))


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
def dxdt(outarray,
         state,
         constants,
         control,
         ref):
    outarray[0] = state[0] + state[ 1]
    outarray[1] = (-state[0] + constants[3]*state[1] + constants[0] * state[2] + constants[7] * ref)
    outarray[2] = (-constants[1] * state[ 2] + constants[2] * state[3]**2)
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


@cuda.jit(void(float64[:,:,::1],
                float64[:,::1],
                float64[::1],
                float64[::1],
                float64,
                int64,
                float64[::1],
                xoro_type[::1],
                float64[::1],
                float64[::1]
                ),
                opt=True)
def naiive_euler_kernel(output,
                        grid_params,
                        constants,
                        inits,
                        duration,
                        output_fs,
                        filtercoeffs,
                        RNG,
                        noise_sigmas,
                        ref):

    #Figure out where we are on the chip
    tx = cuda.threadIdx.x
    block_index = cuda.blockIdx.x
    l_param_set = BLOCKSIZE_X * block_index + tx

    # Don't try and do a run that hasn't been requested.
    if l_param_set >= len(grid_params):
        return

    # store local params in a register - probably done by the compiler regardless
    # but would like to keep memory locations front-of-mind
    l_step_size = constants[-1]
    l_ds_rate = int32(1 / (output_fs * l_step_size))
    l_n_outer = int32((duration / l_step_size) / l_ds_rate)

    # Declare arrays to be kept in shared memory - very quick access.
    s_sums = cuda.shared.array(
        shape=(BLOCKSIZE_X,
               NUM_STATES),
        dtype=float64)

    s_state = cuda.shared.array(
        shape=(BLOCKSIZE_X,
               NUM_STATES),
        dtype=float64)

    # vectorize local variables used in integration for convenience
    l_dxdt = cuda.local.array(
        shape=(NUM_STATES),
        dtype=float64)

    l_noise = cuda.local.array(
        shape=(NUM_STATES),
        dtype=float64)

    l_sigmas = cuda.local.array(
        shape=(NUM_STATES),
        dtype=float64)

    l_RNG = cuda.local.array(
        shape=(NUM_STATES),
        dtype=xoro_type)

    c_filtercoefficients = cuda.const.array_like(filtercoeffs)
    c_filtercoefficients = filtercoeffs
    c_constants = cuda.const.array_like(constants[:9])
    l_a = grid_params[l_param_set, 0]
    l_b = grid_params[l_param_set, 1]
    l_cliplevel = constants[10]
    l_rhat = constants[9]
    l_sigmas = noise_sigmas

    #Initialise w starting states
    for i in range(NUM_STATES):
        s_state[tx, i] = inits[i]
        l_RNG[i] = RNG[l_param_set * NUM_STATES + i]


    s_sums[:] = 0.0
    l_dxdt[:] = 0.0

    #Loop through output samples, one iteration per output
    for i in range(l_n_outer):

        #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
        for j in range(l_ds_rate):

            # Get absolute index of current sample
            abs_sample = i*l_ds_rate + j

            # Generate noise value for each state
            get_noise(l_noise,
                    l_sigmas,
                    l_RNG)


            #Get current filter coefficient for the downsampling filter
            filtercoeff = c_filtercoefficients[j]

            #Clip reference signal (to simulate a hardware limitation)
            ref_i = clip(ref[abs_sample] * l_rhat, l_cliplevel)

            #Clip control signal (to simulate a hardware limitation)
            control = clip(l_a * s_state[tx, 4] + l_b, l_cliplevel)

            # Calculate derivative at sample
            dxdt(l_dxdt,
                s_state[tx],
                c_constants,
                control,
                ref_i)

            for k in range(NUM_STATES):
                s_state[tx, k] += l_dxdt[k] * l_step_size + l_noise[k]
                s_sums[tx, k] += s_state[tx,k] * filtercoeff

        #Grab completed output sample
        output[i, l_param_set, 0] = s_sums[tx,0]
        output[i, l_param_set, 1] = s_sums[tx,1]
        output[i, l_param_set, 2] = s_sums[tx,2]
        output[i, l_param_set, 3] = s_sums[tx,3]
        output[i, l_param_set, 4] = s_sums[tx,4]
        #Reset filters to zero for another run
        s_sums[tx, :] = 0





#Create output array, initialise to 0
outputstates = cuda.pinned_array((int(fs * duration), len(grid_params), NUM_STATES), dtype=np.float64)
outputstates[:, :, :] = 0

#Create reference signal (use as a template for sending your information into the device)
reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

#Convert all arrays to CUDA device arrays, to be passed to the GPU kernel as function arguments
d_outputstates = cuda.to_device(outputstates)
d_constants = cuda.to_device(constants)
d_reference = cuda.to_device(reference)
d_gridparams = cuda.to_device(grid_params)
d_filtercoefficients = cuda.to_device(filtercoefficients)
d_inits = cuda.to_device(inits)

#Create random noise generators (1 per thread)
random_seed = 1
#Indexing note - because we will not add noise to all states, index this like a 2d (state, tx) array
d_noise = create_xoroshiro128p_states(NUMTHREADS*NUM_STATES, random_seed)
d_noisesigmas = cuda.to_device(noise_sigmas)

# Test timing loop
start = cuda.event()
end = cuda.event()
_s = 0
_e = 0
timing_nb = 0
timing_nb_wall = 0
for i in range(4):
    if i > 0:
        start.record()
        _s = time()

    naiive_euler_kernel[BLOCKSPERGRID,
                            BLOCKSIZE_X](
                        d_outputstates,
                        d_gridparams,
                        d_constants,
                        d_inits,
                        duration,
                        fs,
                        d_filtercoefficients,
                        d_noise,
                        d_noisesigmas,
                        d_reference
                        )

    cuda.synchronize()
    print("{} loops done".format(i))
    #Go and get the output array from the GPU
    d_outputstates.copy_to_host(outputstates)
    outputstates = np.asarray(outputstates)
    if i > 0:
        end.record()
        end.synchronize()
        _e = time()
        timing_nb += cuda.event_elapsed_time(start, end)
        timing_nb_wall += (_e - _s)

cuda.profile_stop()
print('numba events:', timing_nb / 3, 'ms')
print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')