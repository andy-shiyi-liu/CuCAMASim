// defined constants for the simulator.
#ifndef CONSTS_H
#define CONSTS_H

// CuCAMASim version number
#define CUCAMASIM_VERSION "1.9.3"

// The GPU device id to be used for the simulation.
#define GPU_DEVICE_ID 0

// RRAM related
//  - newton's method for solving conductance from Vbd
#define RRAM_STARTPOINT 75
#define RRAM_MAX_ITER 100
#define RRAM_TOLERANCE 1e-8
//  - add new mapping on RRAM Conductance
//    - thread and block size used
#define RRAM_NEWMAPPING_THREAD_X 1     // suggested: 16
#define RRAM_NEWMAPPING_THREAD_Y 1     // suggested: 32
//  - add RRAM noise
//    - thread and block size when adding RRAM noise
#define RRAM_NOISE_THREAD_X 16          // suggested: 16
#define RRAM_NOISE_THREAD_Y 32          // suggested: 32

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