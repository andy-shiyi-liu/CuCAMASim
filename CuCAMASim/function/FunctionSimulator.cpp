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

    // 3. add new mapping
    mapping->addNewMapping(camData);

    // 4. add write noise
    writeNoise->addWriteNoise(camData);

    // 5. Data mapping to CAM arrays
    mapping->write(camData);
}