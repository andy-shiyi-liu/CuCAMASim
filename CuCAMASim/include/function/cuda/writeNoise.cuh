#ifndef WRITENOISE_CUH
#define WRITENOISE_CUH

#include "util/data.h"
#include "util/config.h"

#include "function/writeNoise.h"

#include <cstring>

void addRRAMNoise(WriteNoise *writeNoise, ACAMArray *array);

#endif