#include "function/FunctionSimulator.h"

#include <iostream>

#include "function/cuda/distance.cuh"
#include "util/data.h"

void FunctionSimulator::write(CAMData &CAMData){
    CAMData.at(0,0,0) = 0;
    if (camConfig->arrayConfig->cell == "ACAM"){

    }
}