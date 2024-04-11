#include "function/FunctionSimulator.h"

#include <iostream>

#include "function/cuda/distance.cuh"
#include "util/data.h"

void FunctionSimulator::write(CAMData *camData){
    // 1. Quantization (optional for ACAM)
    if (camConfig->arrayConfig->cell != "ACAM"){
        quantizer->write(camData);
    }

    // 2. Conversion to voltage/conductance representation
    converter->write(camData);
}