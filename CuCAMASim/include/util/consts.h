// defined constants for the simulator.
#ifndef CONSTS_H
#define CONSTS_H

// Maximum number of rows that can be matched for a single query.
// This is necessary for allocating GPU memory.
#define MAX_MATCHED_ROWS 10

// thread and block size used in arraySearch()
//  - when calculating distance for every query vector, we launch a 2D grid of
//    blocks, where each block is a 2D grid of threads
#define DIST_FUNC_THREAD_X 1       // for debug, set to 32 after debug
#define DIST_FUNC_THREAD_Y 1       // for debug, set to 32 after debug
//  - when sensing, we launch a 1D grid of blocks, where each block is a 1D grid of threads
#define SENSING_THREAD_X 1          // for debug, set to 1024 after debug

#endif