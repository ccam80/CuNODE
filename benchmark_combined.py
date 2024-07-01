# -*- coding: utf-8 -*-

""" Working array layout

|        | 0                | 1           | 2         | 3          |   | 4                 | 5            | 6         | 7           |   | 8          | 9      | 10       | 11       |
|--------|------------------|-------------|-----------|------------|---|-------------------|--------------|-----------|-------------|---|------------|--------|----------|----------|
| FMA    | op1              | op2         | op3       | res        |   | op1               | op2          | op3       | res         |   | op1        | op2    | op3      | res      |
| 0      | omL*-step        | CTRL        | 0         | ctrl_a     |   | step              | VEL          | DISP      | **disp**    |   | DISP       | FIL    | DISP_SUM | disp_sum |
| 1      | FIL              | VEL         | VEL_SUM   | vel_sum    |   | delta*omega*-step | VEL          | RREF      | vel_a       |   | VELA       | 1      | VELB     | **vel**  |
| 2      | gamma*step       | CTRL2       | TEMP      | gctrl2     |   | beta*-step        | TEMP         | GCTRL2    | **temp**    |   | TEMP       | FIL    | TEMP_SUM | temp_sum |
| 3      | omL*step         | AXB(clip)   | CTRL      | ctrl_b     |   | 1                 | CTRLA        | CTRL_B    | **ctrl**    |   | CTRL       | FIL    | CTRL_SUM | ctrl_sum |
| 4      | rhat*sign*step   | REF         | 0         | rref       |   | step*sign         | VEL          | fhpf      | **HPF**     |   | HPF        | FIL    | HPF_SUM  | HPF_SUM  |
| 5      | (-)step*omega^2  | DISP        | VEL       | dispstep   |   | alpha*step        | TEMP         | DISP_STEP | vel_b       |   | CTRL       | CTRL   | 0        | ctrl2    |
| 6      | omH*-step        | HPF         | HPF       | fhpf       |   |                   |              |           |             |   | a          | HPF    | b        | axb      |
| 7      |                  |             |           |            |   |                   |              |           |             |   |            |        |          |          |

"""

"""benchmarking notes:
    The "optimized" (poorly) version increases linearly with n samples at about 37ms/10000 * n.
    The naiive version is an order of magnitude slower per sample: 37ms/1000 * n.

    The optimized version scales with parralels in blocks between 16384 - 32768.
    Every 16384+ parallel runs cause an extra serial operation, so operation scales with ceil(k*16384)
    The naiive version scales one "block" later: with ceil(k*32768)
    """
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"


from numba import cuda, float64, int64, void, int32, int16, uint64, from_dtype
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
import numpy as np
from filter_coefficients import ds_100_filter
from time import time
from system_parallelisation import working_array_constants, update_indices_constant, fs, duration, step_size, cliplevel, noise_sigmas
xoro_type = from_dtype(xoroshiro128p_dtype)

device_free_memory = cuda.current_context().get_memory_info()[0]

BLOCKSIZE_X = 8         #Threads per solver/operations per param set
BLOCKSIZE_Y = 32        #"cantilevers" per shared memory block. Max SM 64kb so this
                        # should be equal to (64kb - reference_chun/ SM allocation per thread maxish
NUM_EQS = 5
WORKINGARRAY_LENGTH = BLOCKSIZE_X
WORKINGARRAY_WIDTH = 12 + 1 # 4 per math stage -op1, op2, op3, op4, then one extra to lessen bank conflicts (these will still occur between y-coord thread groups - one every two groups I think)
OUTPUTBUFFER_DEPTH = 4

a_gains = np.asarray([i * 0.2 for i in range(-128, 128)], dtype=np.float64)
b_params = np.asarray([i * 0.1 for i in range(-128, 128)], dtype=np.float64)
grid_params = np.asarray([(a, b) for a in a_gains for b in b_params])

#TODO: Consider scaling.
filtercoefficients = np.asarray(ds_100_filter, dtype=np.float64)


# cuda.profile_start()

@cuda.jit(
                void(float64[:,:,::1],
                float64[:,::1],
                float64[:,:,::1],
                int16[:,:,::1],
                float64,
                float64,
                int64,
                float64[::1],
                float64,
                xoro_type[::1],
                float64[::1],
                float64[::1]
                ),
                opt=True)
def cantilever_euler_kernel(output,
                            grid_params,
                            working_constants,
                            update_indices,
                            step_size,
                            duration,
                            output_fs,
                            filtercoeffs,
                            cliplevel,
                            RNG,
                            noise_sigmas,
                            ref):

    #Figure out where we are on the chip
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    block_index = cuda.blockIdx.x
    idx = block_index * ty + ty*tx + tx
    # store local params in a register - probably done by the compiler regardless
    # but would like to keep memory locations front-of-mind
    l_cliplevel = cliplevel
    l_param_set = ty + block_index * BLOCKSIZE_Y
    l_ds_rate = int32(1 / (output_fs * step_size))
    l_n_outer = int32((duration / step_size) / l_ds_rate)

    # Initialise working arrays
    s_updateindices = cuda.shared.array(
            shape=(WORKINGARRAY_WIDTH,
                   WORKINGARRAY_LENGTH,
                   2),
        dtype=int16)

    s_working = cuda.shared.array(
        shape=(BLOCKSIZE_Y,
               WORKINGARRAY_WIDTH,  # Pad to avoid bank conflicts
               WORKINGARRAY_LENGTH),
        dtype=float64)

    for i in range(WORKINGARRAY_WIDTH - 1):  # Ignore the padding row
        s_working[ty, i, tx] = working_constants[tx, i, ty]

    for k in range(2):
        for i in range(WORKINGARRAY_WIDTH - 1):  # Ignore the padding row
            s_updateindices[i, tx, k] = update_indices[tx, i, k]
        s_updateindices[12, tx, 0] = 0
        s_updateindices[12, tx, 1] = 0

    cuda.syncwarp()
    # s_updateindices = update_indices
    # s_working = cuda.shared.array_like(d_working_array_constants)
    s_working[ty, 8, 6] = grid_params[l_param_set, 0]
    s_working[ty, 10, 6] = grid_params[l_param_set, 1]

    c_filtercoefficients = cuda.const.array_like(filtercoeffs)
    c_filtercoefficients = filtercoeffs

    for i in range(l_n_outer):

        for j in range(l_ds_rate):
            thread_noise = xoroshiro128p_normal_float64(RNG, idx) * noise_sigmas[tx]

            #Bring in reference sample, this seems slow. Can I more effectively cache?
            s_working[ty, 1, tx] = cuda.selp(tx == 4, ref[i*l_ds_rate + j], s_working[ty, 1, tx])


            # Update operands using index lookup table
            (sx, sy) = s_updateindices[0, tx]
            s_working[ty, 0, tx] = cuda.selp(sy == 0, s_working[ty, 0, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[1, tx]
            s_working[ty, 1, tx] = cuda.selp(sy == 0, s_working[ty, 1, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[2, tx]
            s_working[ty, 2, tx] = cuda.selp(sy == 0, s_working[ty, 2, tx], s_working[ty, sy, sx])
            cuda.syncwarp()

            s_working[ty, 3, tx] = cuda.fma(s_working[ty, 0, tx], s_working[ty, 1, tx], s_working[ty, 2, tx])
            cuda.syncwarp()

            (sx, sy) = s_updateindices[4, tx]
            s_working[ty, 4, tx] = cuda.selp(sy == 0, s_working[ty, 4, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[5, tx]
            s_working[ty, 5, tx] = cuda.selp(sy == 0, s_working[ty, 5, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[6, tx]
            s_working[ty, 6, tx] = cuda.selp(sy == 0, s_working[ty, 6, tx], s_working[ty, sy, sx])
            cuda.syncwarp()

            s_working[ty, 7, tx] = cuda.fma(s_working[ty, 4, tx], s_working[ty, 5, tx], s_working[ty, 6, tx])
            cuda.syncwarp()

            # Load filter coefficients into table for each time step between steps 1 and 3 (where it is used)
            l_filtercoeff = c_filtercoefficients[j]
            s_working[ty, 9, tx] = cuda.selp(tx in [0, 2, 3, 4], l_filtercoeff, s_working[ty, 9, tx])
            s_working[ty, 0, tx] = cuda.selp(tx == 1, l_filtercoeff, s_working[ty, 0, tx])
            cuda.syncwarp()

            # add noise to states (couldn't cram this in the FMA table)
            s_working[ty, 7, tx] = cuda.selp(tx in [0, 2, 3, 4], s_working[ty, 7, tx] + thread_noise, s_working[ty, 7, tx])
            cuda.syncwarp()

            (sx, sy) = s_updateindices[8, tx]
            s_working[ty, 8, tx] = cuda.selp(sy == 0, s_working[ty, 8, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[9, tx]
            s_working[ty, 9, tx] = cuda.selp(sy == 0, s_working[ty, 9, tx], s_working[ty, sy, sx])
            (sx, sy) = s_updateindices[10, tx]
            s_working[ty, 10, tx] = cuda.selp(sy == 0, s_working[ty, 10, tx], s_working[ty, sy, sx])
            cuda.syncwarp()

            s_working[ty, 11, tx] = cuda.fma(s_working[ty, 8, tx], s_working[ty, 9, tx], s_working[ty, 10, tx])
            cuda.syncwarp()

            # Clip control signal before using it to calculate next step - replicates a hardware limit
            l_control = s_working[ty, 11, 6]
            s_working[ty, 11, 6] = cuda.selp(l_control > l_cliplevel, l_cliplevel, l_control)
            s_working[ty, 11, tx] = cuda.selp(tx == 1, s_working[ty, 11, tx] + thread_noise, s_working[ty, 11, tx]) # add noise to vel while we're selping.
            # Grab the new value of control as otherwise the old one is cached and writtent to a non-neg clipped value
            cuda.syncwarp()
            l_control_posclipped = s_working[ty, 11, 6]
            s_working[ty, 11, 6] = cuda.selp(l_control_posclipped < -l_cliplevel, -l_cliplevel, l_control_posclipped)
            cuda.syncwarp()


        # Note, there is some weirdness here regarding the velocity sum - it is one
        # time step behind the rest, and will be updated on sample = 1. The filter
        # function starts and ends at zero, so maybe no harm done?
        output[i, l_param_set, tx] = cuda.selp(tx == 1, s_working[ty, 3, tx], s_working[ty, 11, tx])
        cuda.syncwarp()

        s_working[ty, 11, tx] = cuda.selp(tx in [0, 2, 3, 4], float64(0.0), s_working[ty, 11, tx])
        s_working[ty, 3, tx] = cuda.selp(tx == 1, float64(0.0), s_working[ty, 3, tx])
        cuda.syncwarp()


durations = np.asarray([1, 10, 100, 1000, 5000])
num_parallel = np.asarray([32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
opt_timings = np.zeros((len(durations), len(num_parallel)))
for j, duration in enumerate(durations):
    for k, parallels in enumerate(num_parallel):
        sampled_combinations = np.random.choice(len(grid_params), size=parallels, replace=(parallels > len(grid_params)))
        sampled_combinations = grid_params[sampled_combinations]

        if (int64(fs)* int64(duration) * (int64(parallels) * int64(BLOCKSIZE_X) + int64(1)) * int64(8)) > (device_free_memory):
            opt_timings[j, k] = 0
        else:
            outputstates = cuda.pinned_array((int(fs * duration), len(sampled_combinations), BLOCKSIZE_X), dtype=np.float64)
            outputstates[:, :, :] = 0
            reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

            NUMTHREADS = parallels * BLOCKSIZE_X
            BLOCKSPERGRID = max(1, parallels // BLOCKSIZE_Y)

            working_array_3d = np.repeat(working_array_constants[:,:,np.newaxis], BLOCKSIZE_Y, axis=2)

            d_outputstates = cuda.to_device(outputstates)
            d_reference = cuda.to_device(reference)
            d_gridparams = cuda.to_device(sampled_combinations)
            d_filtercoefficients = cuda.to_device(filtercoefficients)
            d_workingarray = cuda.to_device(working_array_3d)
            d_updateindices = cuda.to_device(update_indices_constant)

            random_seed = 1
            d_noise = create_xoroshiro128p_states(int(NUMTHREADS), random_seed)
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

                cantilever_euler_kernel[BLOCKSPERGRID,
                                        (BLOCKSIZE_X, BLOCKSIZE_Y)](
                    d_outputstates,
                    d_gridparams,
                    d_workingarray,
                    d_updateindices,
                    step_size,
                    duration,
                    fs,
                    d_filtercoefficients,
                    cliplevel,
                    d_noise,
                    d_noisesigmas,
                    d_reference)

                cuda.synchronize()
                # print("{} loops done".format(i))
                d_outputstates.copy_to_host(outputstates)
                outputstates = np.asarray(outputstates)
                if i > 0:
                    end.record()
                    end.synchronize()
                    _e = time()
                    timing_nb += cuda.event_elapsed_time(start, end)
                    timing_nb_wall += (_e - _s)

            print('run', j*len(num_parallel) + k, ' out of ', len(num_parallel) * len(durations))
            opt_timings[j, k] = timing_nb_wall / 3 * 1000
            # print('numba events:', timing_nb / 3, 'ms')
            print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')


from system_parallelisation import constants, inits

BLOCKSIZE_X = 8 * 32        #Number of param sets per block
NUM_STATES = 5

#Setting up grid of params to simulate with
a_gains = np.asarray([i * 0.2 for i in range(-128, 128)], dtype=np.float64)
b_params = np.asarray([i * 0.1 for i in range(-128, 128)], dtype=np.float64)
grid_params = np.asarray([(a, b) for a in a_gains for b in b_params])


#Bring in filter coefficients for polyphase decimator - FIR filter that downsamples
#TODO: Consider scaling.
filtercoefficients = np.asarray(ds_100_filter, dtype=np.float64)

# This was brought in in an array to save clutter, fetch step size for size calcs
step_size = constants[-1]

NUMTHREADS = len(grid_params)
BLOCKSPERGRID = max(1, len(grid_params) // BLOCKSIZE_X)



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


durations = np.asarray([1, 10, 100, 1000, 5000])
num_parallel = np.asarray([32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
naiive_timings = np.zeros((len(durations), len(num_parallel)))
for j, duration in enumerate(durations):
    for k, parallels in enumerate(num_parallel):
        sampled_combinations = np.random.choice(len(grid_params), size=parallels, replace=(parallels > len(grid_params)))
        sampled_combinations = grid_params[sampled_combinations]

        if (int64(fs) * int64(duration) * (int64(parallels) * int64(NUM_STATES) + int64(1))*8) > (device_free_memory):
            naiive_timings[j, k] = 0
        else:
            NUMTHREADS = len(sampled_combinations)
            BLOCKSPERGRID = int(max(1, np.ceil(len(sampled_combinations) / BLOCKSIZE_X)))

            outputstates = cuda.pinned_array((int(fs * duration), parallels, NUM_STATES), dtype=np.float64)
            outputstates[:, :, :] = 0
            reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

            d_outputstates = cuda.to_device(outputstates)
            d_constants = cuda.to_device(constants)
            d_reference = cuda.to_device(reference)
            d_gridparams = cuda.to_device(sampled_combinations)
            d_filtercoefficients = cuda.to_device(filtercoefficients)
            d_inits = cuda.to_device(inits)

            random_seed = 1
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
                    d_reference)

                cuda.synchronize()
                # print("{} loops done".format(i))
                d_outputstates.copy_to_host(outputstates)
                outputstates = np.asarray(outputstates)
                if i > 0:
                    end.record()
                    end.synchronize()
                    _e = time()
                    timing_nb += cuda.event_elapsed_time(start, end)
                    timing_nb_wall += (_e - _s)

            naiive_timings[j, k] = timing_nb_wall / 3 * 1000
            print('run', j*len(num_parallel) + k, ' out of ', len(num_parallel) * len(durations))
            # print('numba events:', timing_nb / 3, 'ms')
            print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')

deltas = opt_timings - naiive_timings

np.savetxt(f"timing/opt_timings_office_pc_{time()}.txt", opt_timings)
np.savetxt(f"timing/naiive_timings_office_pc_{time()}.txt", naiive_timings)
np.savetxt(f"timing/delta_timings_office_pc_{time()}.txt", deltas)