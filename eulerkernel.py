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

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"


from numba import cuda, float64, int64, void, int32, int16, uint64, from_dtype
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
import numpy as np
from filter_coefficients import ds_100_filter
from time import time
from system_parallelisation import working_array_constants, update_indices_constant, fs, duration, step_size, cliplevel, noise_sigmas


BLOCKSIZE_X = 8         #Threads per solver/operations per param set
BLOCKSIZE_Y = 58        #"cantilevers" per shared memory block. Max SM 64kb so this
                        # should be equal to (64kb - reference_chun/ SM allocation per thread maxish
NUM_EQS = 5
WORKINGARRAY_LENGTH = BLOCKSIZE_X
WORKINGARRAY_WIDTH = 12 + 1 # 4 per math stage -op1, op2, op3, op4, then one extra to lessen bank conflicts (these will still occur between y-coord thread groups - one every two groups I think)
OUTPUTBUFFER_DEPTH = 4

a_gains = np.asarray([i * 0.2 for i in range(-128, 128)], dtype=np.float64)
b_params = np.asarray([i * 0.1 for i in range(-128, 128)], dtype=np.float64)
grid_params = [(a, b) for a in a_gains for b in b_params]

#TODO: Consider scaling.
filtercoefficients = np.asarray(ds_100_filter, dtype=np.float64)

NUMTHREADS = len(grid_params) * BLOCKSIZE_X
BLOCKSPERGRID = max(1, len(grid_params) // BLOCKSIZE_Y)

working_array_3d = np.repeat(working_array_constants[:,:,np.newaxis], BLOCKSIZE_Y, axis=2)
# cuda.profile_start()
xoro_type = from_dtype(xoroshiro128p_dtype)


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




outputstates = cuda.pinned_array((int(fs * duration), len(grid_params), BLOCKSIZE_X), dtype=np.float64)
outputstates[:, :, :] = 0
reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

d_outputstates = cuda.to_device(outputstates)
d_reference = cuda.to_device(reference)
d_gridparams = cuda.to_device(grid_params)
d_filtercoefficients = cuda.to_device(filtercoefficients)
d_workingarray = cuda.to_device(working_array_3d)
d_updateindices = cuda.to_device(update_indices_constant)

random_seed = 1
d_noise = create_xoroshiro128p_states(NUMTHREADS, random_seed)
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
    print("{} loops done".format(i))
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