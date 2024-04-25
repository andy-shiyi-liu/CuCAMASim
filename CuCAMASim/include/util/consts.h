// defined constants for the simulator.
#ifndef CONSTS_H
#define CONSTS_H

// The GPU device id to be used for the simulation.
#define GPU_DEVICE_ID 0

// add RRAM noise
//  - thread and block size when adding RRAM noise
#define RRAM_NOISE_THREAD_X 32
#define RRAM_NOISE_THREAD_Y 32
//  - newton's method for solving conductance from Vbd
#define RRAM_STARTPOINT 75
#define RRAM_MAX_ITER 100
#define RRAM_TOLERANCE 1e-8

// Maximum number of rows that can be matched for a single query.
// This is necessary for allocating GPU memory.
#define MAX_MATCHED_ROWS 10

// thread and block size used in arraySearch()
//  - when calculating distance for every query vector, we launch a 2D grid of
//    blocks, where each block is a 2D grid of threads
#define DIST_FUNC_THREAD_X 32       // for debug, set to 32 after debug
#define DIST_FUNC_THREAD_Y 32       // for debug, set to 32 after debug
//  - when sensing, we launch a 1D grid of blocks, where each block is a 1D grid of threads
#define SENSING_THREAD_X 1024         // for debug, set to 1024 after debug
//  - when merging, we launch a 1D grid of blocks, where each block is a 1D grid of threads
#define MERGING_THREAD_X 1024          // for debug, set to 1024 after debug

#endif