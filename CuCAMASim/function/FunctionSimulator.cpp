#include "function/FunctionSimulator.h"

#include <iostream>

#include "function/cuda/distance.cuh"
#include "util/data.h"

void FunctionSimulator::write(CAMArray *camArray){
    // 1. Quantization (optional for ACAM)
    if (camConfig->arrayConfig->cell != "ACAM"){
        quantizer->write(camArray);
    }

    // 2. Conversion to voltage/conductance representation
    converter->write(camArray);

    // 3. add new mapping
    mapping->addNewMapping(camArray);

    // 4. add write noise
    writeNoise->addWriteNoise(camArray);

    // 5. Data mapping to CAM arrays
    mapping->write(camArray);
}