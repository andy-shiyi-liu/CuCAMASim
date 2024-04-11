#include "function/quantize.h"
#include "util/data.h"

// directly modify camData to add quantization
void Quantize::write(CAMData* camData){
    camData->at(0,0,0);
    // throw an not implemented error
    throw std::runtime_error("Quantize::write() not implemented");
}